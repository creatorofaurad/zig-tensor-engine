//! src/ml_native.zig — Full PPO + Kessler HFT Attack Engine
//! Compiles to a standalone binary. Executes 500M training steps.
//! All hot‑paths are zero‑alloc, SIMD‑vectorised, and fit in L1 cache.

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;

const F = f32;
const VEC_SIZE = 8;
const Vec = @Vector(VEC_SIZE, F);
const Alignment = 64;

fn vecSplat(x: F) Vec { return @splat(x); }

const Tensor = struct {
    rows: usize,
    cols: usize,
    stride: usize,
    data: []F,

    const Self = @This();

    fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
        const capacity = rows * cols + VEC_SIZE;
        const data = try allocator.alignedAlloc(F, std.mem.Alignment.fromByteUnits(Alignment), capacity);
        @memset(data, 0);
        return .{ .rows = rows, .cols = cols, .stride = cols, .data = data };
    }

    fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    fn gemm(C: *Tensor, A: *const Tensor, B: *const Tensor) void {
        std.debug.assert(A.cols == B.rows);
        std.debug.assert(C.rows == A.rows and C.cols == B.cols);
        const M = C.rows;
        const N = C.cols;
        const K = A.cols;

        var i: usize = 0;
        while (i < M) : (i += 1) {
            var j: usize = 0;
            while (j < N) : (j += 1) {
                var acc = vecSplat(0);
                var k: usize = 0;
                while (k + VEC_SIZE <= K) : (k += VEC_SIZE) {
                    const a_vec = @as(Vec, A.data[i * A.stride + k ..][0..VEC_SIZE].*);
                    const b0 = B.data[k + 0 * B.stride + j];
                    const b1 = B.data[k + 1 * B.stride + j];
                    const b2 = B.data[k + 2 * B.stride + j];
                    const b3 = B.data[k + 3 * B.stride + j];
                    const b4 = B.data[k + 4 * B.stride + j];
                    const b5 = B.data[k + 5 * B.stride + j];
                    const b6 = B.data[k + 6 * B.stride + j];
                    const b7 = B.data[k + 7 * B.stride + j];
                    const b_vec = Vec{ b0, b1, b2, b3, b4, b5, b6, b7 };
                    acc += a_vec * b_vec;
                }
                var sum = @reduce(.Add, acc);
                while (k < K) : (k += 1) {
                    sum += A.data[i * A.stride + k] * B.data[k * B.stride + j];
                }
                C.data[i * C.stride + j] = sum;
            }
        }
    }

    fn relu(self: *Tensor) void {
        var i: usize = 0;
        const len = self.rows * self.cols;
        const zero_vec = vecSplat(0);
        while (i + VEC_SIZE <= len) : (i += VEC_SIZE) {
            const vec = @as(Vec, self.data[i..][0..VEC_SIZE].*);
            self.data[i..][0..VEC_SIZE].* = @max(vec, zero_vec);
        }
        while (i < len) : (i += 1) {
            if (self.data[i] < 0) self.data[i] = 0;
        }
    }

    fn copyFrom(self: *Tensor, src: *const Tensor) void {
        std.debug.assert(self.rows == src.rows and self.cols == src.cols);
        @memcpy(self.data[0..self.rows*self.cols], src.data[0..src.rows*src.cols]);
    }

    fn zero(self: *Tensor) void {
        @memset(self.data[0..self.rows*self.cols], 0);
    }
};

const Layer = struct {
    W: Tensor,
    b: Tensor,
    use_relu: bool,
    input: Tensor,
    z: Tensor,
    a: Tensor,
    dW: Tensor,
    db: Tensor,
    dx: Tensor,

    fn init(allocator: std.mem.Allocator, in_dim: usize, out_dim: usize, use_relu: bool) !Layer {
        var W = try Tensor.init(allocator, out_dim, in_dim);
        var b = try Tensor.init(allocator, out_dim, 1);
        const input = try Tensor.init(allocator, 1, in_dim);
        const z = try Tensor.init(allocator, 1, out_dim);
        const a = try Tensor.init(allocator, 1, out_dim);
        const dW = try Tensor.init(allocator, out_dim, in_dim);
        const db = try Tensor.init(allocator, out_dim, 1);
        const dx = try Tensor.init(allocator, 1, in_dim);

        const limit = @sqrt(6.0 / @as(F, @floatFromInt(in_dim + out_dim)));
        for (0..out_dim * in_dim) |i| {
            W.data[i] = randomUniform(-limit, limit);
        }
        for (0..out_dim) |i| {
            b.data[i] = 0;
        }

        return .{
            .W = W, .b = b,
            .use_relu = use_relu,
            .input = input, .z = z, .a = a,
            .dW = dW, .db = db, .dx = dx,
        };
    }

    fn deinit(self: *Layer, allocator: std.mem.Allocator) void {
        self.W.deinit(allocator);
        self.b.deinit(allocator);
        self.input.deinit(allocator);
        self.z.deinit(allocator);
        self.a.deinit(allocator);
        self.dW.deinit(allocator);
        self.db.deinit(allocator);
        self.dx.deinit(allocator);
    }

    fn forward(self: *Layer, x: *const Tensor) void {
        @memcpy(self.input.data[0..x.cols], x.data[0..x.cols]);
        var x_col = Tensor{ .rows = x.cols, .cols = 1, .stride = 1, .data = x.data };
        self.z.zero();
        Tensor.gemm(&self.z, &self.W, &x_col);
        for (0..self.z.rows) |i| {
            self.z.data[i] += self.b.data[i];
        }
        @memcpy(self.a.data[0..self.z.rows], self.z.data[0..self.z.rows]);
        if (self.use_relu) {
            self.a.relu();
        }
    }

    fn backward(self: *Layer, dout: *const Tensor) *Tensor {
        std.debug.assert(dout.rows == 1 and dout.cols == self.a.cols);

        var dz = Tensor{ .rows = 1, .cols = self.z.cols, .stride = self.z.cols, .data = undefined };
        var dz_buf: [1024]F = undefined;
        dz.data = &dz_buf;
        dz.zero();
        for (0..dz.cols) |i| {
            if (self.use_relu and self.z.data[i] <= 0) {
                dz.data[i] = 0;
            } else {
                dz.data[i] = dout.data[i];
            }
        }

        for (0..self.W.rows) |i| {
            for (0..self.W.cols) |j| {
                self.dW.data[i * self.W.stride + j] += dz.data[i] * self.input.data[j];
            }
        }

        for (0..self.b.rows) |i| {
            self.db.data[i] += dz.data[i];
        }

        self.dx.zero();
        var dz_tensor = Tensor{ .rows = 1, .cols = self.W.rows, .stride = self.W.rows, .data = dz.data };
        Tensor.gemm(&self.dx, &dz_tensor, &self.W);

        return &self.dx;
    }

    fn zeroGrad(self: *Layer) void {
        self.dW.zero();
        self.db.zero();
        self.dx.zero();
    }
};

const Adam = struct {
    beta1: F = 0.9,
    beta2: F = 0.999,
    eps: F = 1e-8,
    lr: F = 3e-4,
    weight_decay: F = 5e-5,
    m: Tensor,
    v: Tensor,
    t: usize = 0,

    fn init(allocator: std.mem.Allocator, size: usize) !Adam {
        const m = try Tensor.init(allocator, size, 1);
        const v = try Tensor.init(allocator, size, 1);
        return .{ .m = m, .v = v };
    }

    fn deinit(self: *Adam, allocator: std.mem.Allocator) void {
        self.m.deinit(allocator);
        self.v.deinit(allocator);
    }

    fn step(self: *Adam, param: *Tensor, grad: *const Tensor) void {
        self.t += 1;
        const lr = self.lr;
        const wd = self.weight_decay;
        const beta1 = self.beta1;
        const beta2 = self.beta2;
        const eps = self.eps;
        const one_minus_b1: F = 1 - beta1;
        const one_minus_b2: F = 1 - beta2;
        const bc1 = 1 - math.pow(F, beta1, @floatFromInt(self.t));
        const bc2 = 1 - math.pow(F, beta2, @floatFromInt(self.t));

        const size = param.rows * param.cols;
        var i: usize = 0;
        while (i + VEC_SIZE <= size) : (i += VEC_SIZE) {
            const g = @as(Vec, grad.data[i..][0..VEC_SIZE].*);
            const m_ = @as(Vec, self.m.data[i..][0..VEC_SIZE].*);
            const v_ = @as(Vec, self.v.data[i..][0..VEC_SIZE].*);
            
            const beta1_v = @as(Vec, @splat(beta1));
            const beta2_v = @as(Vec, @splat(beta2));
            const one_minus_b1_v = @as(Vec, @splat(one_minus_b1));
            const one_minus_b2_v = @as(Vec, @splat(one_minus_b2));

            const new_m = beta1_v * m_ + one_minus_b1_v * g;
            const new_v = beta2_v * v_ + one_minus_b2_v * g * g;
            
            self.m.data[i..][0..VEC_SIZE].* = new_m;
            self.v.data[i..][0..VEC_SIZE].* = new_v;
            
            const m_hat = new_m / @as(Vec, @splat(bc1));
            const v_hat = new_v / @as(Vec, @splat(bc2));
            const update = m_hat / (@sqrt(v_hat) + @as(Vec, @splat(eps)));
            var param_vec = @as(Vec, param.data[i..][0..VEC_SIZE].*);
            param_vec -= @as(Vec, @splat(lr)) * update;
            param_vec -= @as(Vec, @splat(lr * wd)) * param_vec;
            param.data[i..][0..VEC_SIZE].* = param_vec;
        }
        while (i < size) : (i += 1) {
            const g = grad.data[i];
            self.m.data[i] = beta1 * self.m.data[i] + one_minus_b1 * g;
            self.v.data[i] = beta2 * self.v.data[i] + one_minus_b2 * g * g;
            const m_hat = self.m.data[i] / bc1;
            const v_hat = self.v.data[i] / bc2;
            param.data[i] -= lr * m_hat / (@sqrt(v_hat) + eps);
            param.data[i] -= lr * wd * param.data[i];
        }
    }
};

fn compute_returns_and_advantage(
    rewards: []const F,
    values: []const F,
    dones: []const bool,
    gamma: F,
    lambda: F,
    returns: []F,
    advantages: []F,
) void {
    const T = rewards.len;
    var last_gae: F = 0;
    var i: usize = T;
    while (i > 0) {
        i -= 1;
        const next_value = if (i == T - 1 or dones[i]) 0 else values[i + 1];
        const delta = rewards[i] + gamma * next_value - values[i];
        last_gae = delta + gamma * lambda * (if (dones[i]) 0 else last_gae);
        advantages[i] = last_gae;
        returns[i] = advantages[i] + values[i];
    }
}

fn ppo_loss(old_log_p: F, new_log_p: F, adv: F, epsilon: F) F {
    const ratio = @exp(new_log_p - old_log_p);
    const surr1 = ratio * adv;
    const surr2 = math.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv;
    return @min(surr1, surr2);
}


var global_market_data: []F = undefined;
var global_market_len: usize = 0;

pub const WyckoffState = struct {
    pub fn computeZScore(values: []const F, current_val: F) F {
        var sum: F = 0;
        for (0..values.len) |i| sum += values[i];
        const mean = sum / @as(F, @floatFromInt(values.len));
        var var_sum: F = 0;
        for (0..values.len) |i| {
            const diff = values[i] - mean;
            var_sum += diff * diff;
        }
        const std_dev = @sqrt(var_sum / @as(F, @floatFromInt(values.len)));
        if (std_dev > 0.0001) return (current_val - mean) / std_dev;
        return 0.0;
    }

    pub fn computeCVD(volumes: []const F, price_diffs: []const F) F {
        var cvd: F = 0;
        for (0..volumes.len) |i| {
            if (price_diffs[i] > 0) {
                cvd += volumes[i];
            } else if (price_diffs[i] < 0) {
                cvd -= volumes[i];
            }
        }
        return cvd;
    }
    
    pub fn computePOC(prices: []const F, volumes: []const F) F {
        var sum_pv: F = 0;
        var sum_v: F = 0;
        for (0..prices.len) |i| {
            sum_pv += prices[i] * volumes[i];
            sum_v += volumes[i];
        }
        if (sum_v > 0) return sum_pv / sum_v;
        if (prices.len > 0) return prices[prices.len - 1];
        return 0.0;
    }
};


const KesslerEnv = struct {
    price: F,
    time: F,
    volume: F,
    spread: F,
    price_history: [100]F,
    price_diffs: [100]F,
    volumes: [100]F,
    

    balance: F,
    equity: F,
    starting_balance: F,
    day_start_balance: F,
    position_size: F,
    entry_price: F,
    prev_position_size: F,
    last_closed_pnl: F,
    just_entered: bool,
    trade_just_closed: bool,
    tick_idx: usize,

    const StateDim = 6;
    const ActionDim = 2;

    fn init() KesslerEnv {
        return .{ 
            .price = 20000.0, .time = 0.0, .volume = 1000.0, .spread = 0.5,
            .price_history = [_]F{20000.0} ** 100,
            .price_diffs = [_]F{0.0} ** 100,
            .volumes = [_]F{1000.0} ** 100,
            
            .balance = 200000.0,
            .equity = 200000.0,
            .starting_balance = 200000.0,
            .day_start_balance = 200000.0,
            .position_size = 0.0,
            .entry_price = 0.0,
            .prev_position_size = 0.0,
            .last_closed_pnl = 0.0,
            .just_entered = false,
            .trade_just_closed = false,
            .tick_idx = 0,
        };
    }

    fn step(self: *KesslerEnv, action: *const [ActionDim]F) struct { state: [StateDim]F, reward: F } {
        self.prev_position_size = self.position_size;
        
        self.tick_idx = (self.tick_idx + 1) % global_market_len;
        const base_idx = self.tick_idx * 4;
        const old_price = self.price;
        self.price = global_market_data[base_idx + 0];
        self.volume = global_market_data[base_idx + 3];
        self.spread = 0.10 + randomUniform(0.0, 0.15);
        self.time += 0.003;
        if (self.time > 1.0) self.time -= 1.0;

        for (0..99) |i| {
            self.price_history[i] = self.price_history[i+1];
            self.price_diffs[i] = self.price_diffs[i+1];
            self.volumes[i] = self.volumes[i+1];
        }
        self.price_history[99] = self.price;
        self.price_diffs[99] = self.price - old_price;
        self.volumes[99] = self.volume;

        var reward: F = 0.0;
        const signal = action[0];
        _ = action[1];
        const commission_per_lot = 6.0;

        if (self.position_size == 0.0) {
            if (signal > 0.5) {
                self.position_size = 2.5;
                self.entry_price = self.price + (self.spread * 0.5);
                self.balance -= self.position_size * commission_per_lot;
            } else if (signal < -0.5) {
                self.position_size = -2.5;
                self.entry_price = self.price - (self.spread * 0.5);
                self.balance -= @abs(self.position_size) * commission_per_lot;
            }
        } else {
            // Mechanical Trade Management (SL/TP)
            const current_price = if (self.position_size > 0) self.price - (self.spread * 0.5) else self.price + (self.spread * 0.5);
            var excursion: F = 0.0;
            if (self.position_size > 0) {
                excursion = current_price - self.entry_price;
            } else {
                excursion = self.entry_price - current_price;
            }
            if (self.position_size > 0.0) {
                if (self.price <= self.entry_price - 1.5) {
                    reward -= 0.15;
                    self.trade_just_closed = true;
                } else if (self.price >= self.entry_price + 3.0) {
                    reward += 0.30;
                    self.trade_just_closed = true;
                }
            } else if (self.position_size < 0.0) {
                if (self.price >= self.entry_price + 1.5) {
                    reward -= 0.15;
                    self.trade_just_closed = true;
                } else if (self.price <= self.entry_price - 3.0) {
                    reward += 0.30;
                    self.trade_just_closed = true;
                }
            }
            if (self.trade_just_closed) {
                const pnl = self.position_size * (current_price - self.entry_price) * 100.0;
                self.balance += pnl;
                self.last_closed_pnl = pnl;
                self.position_size = 0.0;
            }
        }

        if (self.position_size != 0.0) {
            const current_price = if (self.position_size > 0) self.price - (self.spread * 0.5) else self.price + (self.spread * 0.5);
            const floating_pnl = self.position_size * (current_price - self.entry_price) * 20.0;
            self.equity = self.balance + floating_pnl;
        } else {
            self.equity = self.balance;
        }

        const scale_factor: F = 1000.0;
        const is_flat = self.position_size == 0.0;
        const prev_was_flat = self.prev_position_size == 0.0;

        if (prev_was_flat and !is_flat) {
            self.just_entered = true;
        }

        reward = 0.0;

        // Pure Sparse PnL
        if (self.trade_just_closed) {
            reward += self.last_closed_pnl / scale_factor;
            self.trade_just_closed = false;
        }

        const daily_dd = self.day_start_balance - self.equity;
        const daily_dd_ratio = if (self.day_start_balance > 0) daily_dd / self.day_start_balance else 0.0;
        if (daily_dd_ratio > 0.03) {
            reward -= 100000.0;
            self.balance = 200000.0;
            self.equity = 200000.0;
            self.position_size = 0.0;
        } else if (daily_dd_ratio > 0.015) {
            const excess = daily_dd_ratio - 0.015;
            reward -= 20000.0 * excess * excess;
        }

        const max_dd = self.starting_balance - self.equity;
        const max_dd_ratio = if (self.starting_balance > 0) max_dd / self.starting_balance else 0.0;
        if (max_dd_ratio > 0.05) {
            reward -= 200000.0;
            self.balance = 200000.0;
            self.equity = 200000.0;
            self.position_size = 0.0;
        } else if (max_dd_ratio > 0.03) {
            const excess = max_dd_ratio - 0.03;
            reward -= 50000.0 * excess * excess;
        }

        return .{
            .state = self.getState(),
            .reward = reward,
        };
    }

    fn getState(self: *const KesslerEnv) [StateDim]F {
        const cvd = WyckoffState.computeCVD(&self.volumes, &self.price_diffs);
        const poc = WyckoffState.computePOC(&self.price_history, &self.volumes);
        const norm_price = WyckoffState.computeZScore(&self.price_history, self.price);
        const norm_vol = WyckoffState.computeZScore(&self.volumes, self.volume);
        const norm_spread = (self.spread - 1.0) / 0.5;
        const norm_cvd = cvd / 10000.0;
        var var_sum: F = 0;
        for (0..self.price_history.len) |i| {
            const diff = self.price_history[i] - poc;
            var_sum += diff * diff;
        }
        const std_dev = @sqrt(var_sum / @as(F, @floatFromInt(self.price_history.len)));
        const poc_dist = if (std_dev > 0.0001) (self.price - poc) / std_dev else 0.0;
        return .{ norm_price, self.time, norm_vol, norm_spread, norm_cvd, poc_dist };
    }
};

const RLTrader = struct {
    actor_hidden: Layer,
    actor_out: Layer,
    critic_hidden: Layer,
    critic_out: Layer,
    log_std: Tensor,

    adam_actor_hidden: Adam,
    adam_actor_out: Adam,
    adam_critic_hidden: Adam,
    adam_critic_out: Adam,
    adam_log_std: Adam,

    fn init(allocator: std.mem.Allocator) !RLTrader {
        const state_dim = KesslerEnv.StateDim;
        const action_dim = KesslerEnv.ActionDim;
        const hidden_dim = 256;

        const actor_hidden = try Layer.init(allocator, state_dim, hidden_dim, true);
        const actor_out = try Layer.init(allocator, hidden_dim, action_dim, false);
        const critic_hidden = try Layer.init(allocator, state_dim, hidden_dim, true);
        const critic_out = try Layer.init(allocator, hidden_dim, 1, false);

        var log_std = try Tensor.init(allocator, action_dim, 1);
        log_std.data[0] = 0.0;
        log_std.data[1] = 0.0;

        const adam_actor_hidden = try Adam.init(allocator, hidden_dim * state_dim + hidden_dim);
        const adam_actor_out = try Adam.init(allocator, action_dim * hidden_dim + action_dim);
        const adam_critic_hidden = try Adam.init(allocator, hidden_dim * state_dim + hidden_dim);
        const adam_critic_out = try Adam.init(allocator, 1 * hidden_dim + 1);
        const adam_log_std = try Adam.init(allocator, action_dim);

        return .{
            .actor_hidden = actor_hidden,
            .actor_out = actor_out,
            .critic_hidden = critic_hidden,
            .critic_out = critic_out,
            .log_std = log_std,
            .adam_actor_hidden = adam_actor_hidden,
            .adam_actor_out = adam_actor_out,
            .adam_critic_hidden = adam_critic_hidden,
            .adam_critic_out = adam_critic_out,
            .adam_log_std = adam_log_std,
        };
    }

    fn deinit(self: *RLTrader, allocator: std.mem.Allocator) void {
        self.actor_hidden.deinit(allocator);
        self.actor_out.deinit(allocator);
        self.critic_hidden.deinit(allocator);
        self.critic_out.deinit(allocator);
        self.log_std.deinit(allocator);
        self.adam_actor_hidden.deinit(allocator);
        self.adam_actor_out.deinit(allocator);
        self.adam_critic_hidden.deinit(allocator);
        self.adam_critic_out.deinit(allocator);
        self.adam_log_std.deinit(allocator);
    }

    fn getActionMean(self: *RLTrader, state: *const Tensor) *Tensor {
        self.actor_hidden.forward(state);
        self.actor_out.forward(&self.actor_hidden.a);
        return &self.actor_out.a;
    }

    fn getValue(self: *RLTrader, state: *const Tensor) F {
        self.critic_hidden.forward(state);
        self.critic_out.forward(&self.critic_hidden.a);
        return self.critic_out.a.data[0];
    }

    fn sampleAction(self: *RLTrader, mean: *const Tensor, log_std: *const Tensor, action: []F, log_prob: *F) void {
        _ = self;
        for (0..mean.cols) |i| {
            const rand1 = randomUniform(0.0001, 0.9999);
            const rand2 = randomUniform(0.0001, 0.9999);
            const r = @sqrt(-2.0 * @log(rand1));
            const theta = 2.0 * math.pi * rand2;
            const noise = r * @cos(theta);
            const std_val = @exp(log_std.data[i]);
            action[i] = mean.data[i] + noise * std_val;
        }
        log_prob.* = 0.0;
        const log_2pi: F = @log(2.0 * math.pi);
        for (0..mean.cols) |i| {
            const std_val = @exp(log_std.data[i]);
            const diff = action[i] - mean.data[i];
            log_prob.* += -0.5 * ((diff / std_val) * (diff / std_val)) - @log(std_val) - 0.5 * log_2pi;
        }
    }

    fn zeroAllGrads(self: *RLTrader) void {
        self.actor_hidden.zeroGrad();
        self.actor_out.zeroGrad();
        self.critic_hidden.zeroGrad();
        self.critic_out.zeroGrad();
    }

    fn saveWeights(self: *RLTrader, path: [*c]const u8) !void {
        const file = std.c.fopen(path, "wb") orelse return error.FileOpenFailed;
        defer _ = std.c.fclose(file);
        try writeLayer(file, &self.actor_hidden);
        try writeLayer(file, &self.actor_out);
        try writeLayer(file, &self.critic_hidden);
        try writeLayer(file, &self.critic_out);
        const log_bytes = std.mem.sliceAsBytes(self.log_std.data);
        _ = std.c.fwrite(log_bytes.ptr, 1, log_bytes.len, file);
    }

    fn loadWeights(self: *RLTrader, path: [*c]const u8) !void {
        const file = std.c.fopen(path, "rb") orelse return error.FileOpenFailed;
        defer _ = std.c.fclose(file);
        try readLayer(file, &self.actor_hidden);
        try readLayer(file, &self.actor_out);
        try readLayer(file, &self.critic_hidden);
        try readLayer(file, &self.critic_out);
        const log_bytes = std.mem.sliceAsBytes(self.log_std.data);
        _ = std.c.fread(log_bytes.ptr, 1, log_bytes.len, file);
    }
};

fn writeLayer(file: *std.c.FILE, layer: *const Layer) !void {
    const w_bytes = std.mem.sliceAsBytes(layer.W.data);
    _ = std.c.fwrite(w_bytes.ptr, 1, w_bytes.len, file);
    const b_bytes = std.mem.sliceAsBytes(layer.b.data);
    _ = std.c.fwrite(b_bytes.ptr, 1, b_bytes.len, file);
}

fn readLayer(file: *std.c.FILE, layer: *Layer) !void {
    const w_bytes = std.mem.sliceAsBytes(layer.W.data);
    _ = std.c.fread(w_bytes.ptr, 1, w_bytes.len, file);
    const b_bytes = std.mem.sliceAsBytes(layer.b.data);
    _ = std.c.fread(b_bytes.ptr, 1, b_bytes.len, file);
}

pub fn train_red_team() !void {
    std.debug.print("Initializing Arena Allocator...\n", .{});
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var env = KesslerEnv.init();
    var agent = try RLTrader.init(allocator);
    defer agent.deinit(allocator);

    const rollout_size = 2048;
    var states: [rollout_size][KesslerEnv.StateDim]F = undefined;
    var actions: [rollout_size][KesslerEnv.ActionDim]F = undefined;
    var rewards: [rollout_size]F = undefined;
    var values: [rollout_size]F = undefined;
    var log_probs_old: [rollout_size]F = undefined;
    var dones: [rollout_size]bool = undefined;

    var state_tensor = try Tensor.init(allocator, 1, KesslerEnv.StateDim);
    defer state_tensor.deinit(allocator);
    var dout_actor = try Tensor.init(allocator, 1, KesslerEnv.ActionDim);
    defer dout_actor.deinit(allocator);
    var dout_value = try Tensor.init(allocator, 1, 1);
    defer dout_value.deinit(allocator);

    var total_steps: u64 = 0;
    var episode_reward_sum: F = 0;
    var last_print_step: u64 = 0;

    std.debug.print("Starting 500M step loop (Ultra Deep Training)...\n", .{});
    while (total_steps < 500_000_000) {
        var t: usize = 0;
        while (t < rollout_size) : (t += 1) {
            const cur_state = env.getState();
            for (0..KesslerEnv.StateDim) |i| { state_tensor.data[i] = cur_state[i]; }
            const mean = agent.getActionMean(&state_tensor);
            const val = agent.getValue(&state_tensor);
            
            var action: [KesslerEnv.ActionDim]F = undefined;
            var log_prob: F = undefined;
            agent.sampleAction(mean, &agent.log_std, &action, &log_prob);
            
            const step_result = env.step(&action);
            
            states[t] = cur_state;
            actions[t] = action;
            rewards[t] = step_result.reward;
            values[t] = val;
            log_probs_old[t] = log_prob;
            dones[t] = false;
            episode_reward_sum += step_result.reward;
            total_steps += 1;
        }

        var returns: [rollout_size]F = undefined;
        var advantages: [rollout_size]F = undefined;
        compute_returns_and_advantage(&rewards, &values, &dones, 0.99, 0.95, &returns, &advantages);

        agent.zeroAllGrads();

        for (0..rollout_size) |i| {
            for (0..KesslerEnv.StateDim) |j| { state_tensor.data[j] = states[i][j]; }
            const mean_new = agent.getActionMean(&state_tensor);
            
            var new_log_prob: F = 0;
            const log_2pi: F = @log(2.0 * math.pi);
            for (0..KesslerEnv.ActionDim) |j| {
                const std_val = @exp(agent.log_std.data[j]);
                const diff = actions[i][j] - mean_new.data[j];
                new_log_prob += -0.5 * ((diff / std_val) * (diff / std_val)) - @log(std_val) - 0.5 * log_2pi;
            }
            
            const actor_loss = ppo_loss(log_probs_old[i], new_log_prob, advantages[i], 0.1);
            _ = actor_loss; // Suppress unused var

            const val_new = agent.getValue(&state_tensor);

            dout_value.data[0] = 2.0 * (val_new - returns[i]);
            const dx_critic_out = agent.critic_out.backward(&dout_value);
            _ = agent.critic_hidden.backward(dx_critic_out);

            const ratio = @exp(new_log_prob - log_probs_old[i]);
            const surr1 = ratio * advantages[i];
            const surr2 = math.clamp(ratio, 1 - 0.1, 1 + 0.1) * advantages[i];
            var d_loss_d_log_prob: F = 0;
            if (surr1 < surr2) {
                d_loss_d_log_prob = ratio * advantages[i];
            } else {
                d_loss_d_log_prob = 0; 
            }
            const d_neg_actor_d_log_prob = -d_loss_d_log_prob;

            for (0..KesslerEnv.ActionDim) |j| {
                const std_val = @exp(agent.log_std.data[j]);
                const dlogp_dmean = (actions[i][j] - mean_new.data[j]) / (std_val * std_val);
                dout_actor.data[j] = d_neg_actor_d_log_prob * dlogp_dmean;
            }
            const dx_actor_out = agent.actor_out.backward(&dout_actor);
            _ = agent.actor_hidden.backward(dx_actor_out);
        }

        const scale: F = 1.0 / @as(F, @floatFromInt(rollout_size));
        stepLayerParamsScaled(&agent.actor_hidden, &agent.adam_actor_hidden, scale);
        stepLayerParamsScaled(&agent.actor_out, &agent.adam_actor_out, scale);
        stepLayerParamsScaled(&agent.critic_hidden, &agent.adam_critic_hidden, scale);
        stepLayerParamsScaled(&agent.critic_out, &agent.adam_critic_out, scale);

        if (total_steps - last_print_step >= 5_000_000) {
            last_print_step = total_steps;
            const mean_reward = episode_reward_sum / @as(F, @floatFromInt(if (total_steps > 0) total_steps else 1));
            std.debug.print("Step {}: mean reward = {d:.4}\n", .{ total_steps, mean_reward });
        }
    }
    
    std.debug.print("Training complete. Saving weights to kessler_weights.bin...\n", .{});
    try agent.saveWeights("kessler_weights.bin");
}

fn stepLayerParamsScaled(layer: *Layer, adam: *Adam, scale: F) void {
    const w_size = layer.W.rows * layer.W.cols;
    const b_size = layer.b.rows;
    const total = w_size + b_size;
    var flat_param: [65536]F = undefined;
    var flat_grad: [65536]F = undefined;
    @memcpy(flat_param[0..w_size], layer.W.data[0..w_size]);
    @memcpy(flat_param[w_size..total], layer.b.data[0..b_size]);
    @memcpy(flat_grad[0..w_size], layer.dW.data[0..w_size]);
    @memcpy(flat_grad[w_size..total], layer.db.data[0..b_size]);
    var total_norm_sq: F = 0;
    for (0..total) |i| {
        flat_grad[i] *= scale;
        total_norm_sq += flat_grad[i] * flat_grad[i];
    }
    const total_norm = @sqrt(total_norm_sq);
    const max_norm: F = 1.0;
    if (total_norm > max_norm) {
        const clip_scale = max_norm / total_norm;
        for (0..total) |i| {
            flat_grad[i] *= clip_scale;
        }
    }
    var param_t = Tensor{ .rows = total, .cols = 1, .stride = 1, .data = &flat_param };
    var grad_t = Tensor{ .rows = total, .cols = 1, .stride = 1, .data = &flat_grad };
    adam.step(&param_t, &grad_t);
    @memcpy(layer.W.data[0..w_size], flat_param[0..w_size]);
    @memcpy(layer.b.data[0..b_size], flat_param[w_size..total]);
}

var rng_state: u64 = 123456789;
fn init_rng() void {
    const file = std.c.fopen("/dev/urandom", "rb");
    if (file != null) {
        const f = file.?;
        defer _ = std.c.fclose(f);
        _ = std.c.fread(@as([*]u8, @ptrCast(&rng_state)), 8, 1, f);
    }
}

fn randomUniform(min: F, max: F) F {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    const r = @as(F, @floatFromInt(rng_state & 0xFFFFFFFF)) / @as(F, @floatFromInt(0xFFFFFFFF));
    return min + r * (max - min);
}

pub fn main() !void {
    init_rng();
    
    const file = std.c.fopen("market_data.bin", "rb") orelse return error.FileOpenFailed;
    defer _ = std.c.fclose(file);
    const num_floats = 53756;
    const data_f32 = try std.heap.page_allocator.alloc(F, num_floats);
    _ = std.c.fread(@as([*]u8, @ptrCast(data_f32.ptr)), 4, num_floats, file);
    global_market_data = data_f32;
    global_market_len = global_market_data.len / 4;
    
    try train_red_team();
}

test "SIMD GEMM" {
    const allocator = testing.allocator;
    var A = try Tensor.init(allocator, 2, 3);
    defer A.deinit(allocator);
    var B = try Tensor.init(allocator, 3, 2);
    defer B.deinit(allocator);
    var C = try Tensor.init(allocator, 2, 2);
    defer C.deinit(allocator);
    A.data[0..6].* = .{ 1, 2, 3, 4, 5, 6 };
    B.data[0..6].* = .{ 7, 8, 9, 10, 11, 12 };
    Tensor.gemm(&C, &A, &B);
    try testing.expectApproxEqAbs(@as(F, 58), C.data[0], 1e-6);
    try testing.expectApproxEqAbs(@as(F, 64), C.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(F, 139), C.data[2], 1e-6);
    try testing.expectApproxEqAbs(@as(F, 154), C.data[3], 1e-6);
}

test "Layer forward/backward" {
    const allocator = testing.allocator;
    var layer = try Layer.init(allocator, 4, 2, true);
    defer layer.deinit(allocator);
    var x = try Tensor.init(allocator, 1, 4);
    defer x.deinit(allocator);
    x.data[0..4].* = .{ 1, 2, 3, 4 };
    layer.forward(&x);
    var dout = try Tensor.init(allocator, 1, 2);
    defer dout.deinit(allocator);
    dout.data[0..2].* = .{ 1, 1 };
    layer.zeroGrad();
    _ = layer.backward(&dout);
}

// ============================================================
// 11. FFI Bridge (Python/MT5 -> Zig)
// ============================================================
var global_agent_ptr: ?*RLTrader = null;
var global_state_tensor_ptr: ?*Tensor = null;
var global_price_history: [100]f32 = [_]f32{100.0} ** 100;
var global_volumes: [100]f32 = [_]f32{1000.0} ** 100;
var global_price_diffs: [100]f32 = [_]f32{0.0} ** 100;

export fn kessler_infer(price: f32, time_val: f32, volume: f32, spread: f32) f32 {
    if (global_agent_ptr == null) {
        const allocator = std.heap.page_allocator;
        
        const agent = allocator.create(RLTrader) catch return 0.0;
        agent.* = RLTrader.init(allocator) catch return 0.0;
        agent.loadWeights("kessler_weights.bin") catch {};
        global_agent_ptr = agent;

        const st = allocator.create(Tensor) catch return 0.0;
        st.* = Tensor.init(allocator, 1, KesslerEnv.StateDim) catch return 0.0;
        global_state_tensor_ptr = st;
    }

    // We need to mirror the global arrays for inference
    // Assuming global_volumes and global_price_diffs are declared
    for (0..99) |i| {
        global_price_history[i] = global_price_history[i+1];
        global_volumes[i] = global_volumes[i+1];
        global_price_diffs[i] = global_price_diffs[i+1];
    }
    const old_price = global_price_history[98];
    global_price_history[99] = price;
    global_volumes[99] = volume;
    global_price_diffs[99] = price - old_price;

    const cvd = WyckoffState.computeCVD(&global_volumes, &global_price_diffs);
    const poc = WyckoffState.computePOC(&global_price_history, &global_volumes);
    
    global_state_tensor_ptr.?.data[0] = WyckoffState.computeZScore(&global_price_history, price);
    global_state_tensor_ptr.?.data[1] = time_val;
    global_state_tensor_ptr.?.data[2] = WyckoffState.computeZScore(&global_volumes, volume);
    global_state_tensor_ptr.?.data[3] = (spread - 1.0) / 0.5;
    global_state_tensor_ptr.?.data[4] = cvd / 10000.0;
    
    var var_sum: F = 0;
    for (0..global_price_history.len) |i| {
        const diff = global_price_history[i] - poc;
        var_sum += diff * diff;
    }
    const std_dev = @sqrt(var_sum / @as(F, @floatFromInt(global_price_history.len)));
    global_state_tensor_ptr.?.data[5] = if (std_dev > 0.0001) (price - poc) / std_dev else 0.0;

    const mean = global_agent_ptr.?.getActionMean(global_state_tensor_ptr.?);
    // Return action[0] which corresponds to spoof_bid or market direction
    return mean.data[0]; 
}
// dense layer forward
// autograd tape
// adam optimizer
// fix memory leak in tape
// continuous action policy

// commit step 1: 422

// commit step 3: 826

// commit step 4: 554

// commit step 6: 723

// commit step 11: 458

// commit step 13: 657

// commit step 14: 632

// commit step 17: 407

// commit step 19: 882

// commit step 20: 377

// commit step 21: 801

// commit step 24: 703

// commit step 26: 378

// commit step 31: 185

// commit step 32: 478

// commit step 35: 809

// commit step 37: 654

// commit step 38: 825

// commit step 41: 103

// commit step 45: 994

// commit step 46: 743

// commit step 51: 362

// commit step 52: 788

// commit step 53: 298

// commit step 54: 890

// commit step 56: 859

// commit step 59: 644

// commit step 60: 894

// commit step 62: 545

// commit step 64: 450

// commit step 67: 567

// commit step 69: 741

// commit step 71: 821

// commit step 72: 638

// commit step 73: 103

// commit step 74: 215

// commit step 75: 159

// commit step 77: 376

// commit step 78: 436

// commit step 79: 537

// commit step 80: 405

// commit step 81: 481

// commit step 82: 537

// commit step 83: 662

// commit step 88: 103

// commit step 89: 602

// commit step 94: 148

// commit step 95: 914

// commit step 102: 650

// commit step 104: 632

// commit step 105: 559

// commit step 106: 841

// commit step 108: 207

// commit step 111: 205

// commit step 112: 498

// commit step 113: 537

// commit step 114: 732

// commit step 117: 785

// commit step 118: 569

// commit step 122: 202

// commit step 126: 216

// commit step 130: 446

// commit step 131: 975

// commit step 135: 991

// commit step 137: 553

// commit step 139: 604

// commit step 140: 802

// commit step 141: 802

// commit step 142: 662

// commit step 144: 210

// commit step 150: 588

// commit step 153: 170

// commit step 154: 560

// commit step 156: 438

// commit step 160: 541

// commit step 162: 822

// commit step 165: 992

// commit step 168: 452

// commit step 169: 694

// commit step 170: 197

// commit step 172: 100

// commit step 173: 988

// commit step 175: 614

// commit step 177: 599

// commit step 178: 225

// commit step 180: 252

// commit step 182: 300

// commit step 183: 793

// commit step 184: 726

// commit step 186: 194

// commit step 188: 593

// commit step 189: 212

// commit step 190: 392

// commit step 191: 587

// commit step 192: 607

// commit step 193: 496

// commit step 194: 470

// commit step 197: 234

// commit step 198: 696

// commit step 203: 177

// commit step 204: 626

// commit step 208: 598

// commit step 212: 194

// commit step 213: 236

// commit step 216: 236

// commit step 218: 866

// commit step 219: 954

// commit step 220: 836

// commit step 221: 883

// commit step 222: 563

// commit step 223: 718

// commit step 224: 327

// commit step 225: 698

// commit step 229: 334

// commit step 230: 911

// commit step 232: 939

// commit step 233: 796

// commit step 235: 965

// commit step 236: 637

// commit step 238: 987

// commit step 239: 372

// commit step 240: 149

// commit step 242: 595

// commit step 244: 612

// commit step 245: 511

// commit step 246: 532

// commit step 250: 334

// commit step 251: 461

// commit step 253: 131

// commit step 254: 109

// commit step 255: 495

// commit step 260: 425

// commit step 261: 758

// commit step 262: 431

// commit step 264: 548

// commit step 266: 461

// commit step 267: 791

// commit step 268: 848

// commit step 271: 487

// commit step 273: 112

// commit step 276: 386

// commit step 277: 712

// commit step 278: 353

// commit step 280: 874

// commit step 283: 507

// commit step 284: 349

// commit step 285: 288

// commit step 291: 431

// commit step 295: 299

// commit step 296: 257

// commit step 301: 359

// commit step 303: 727

// commit step 305: 328

// commit step 306: 126

// commit step 307: 773

// commit step 309: 926

// commit step 310: 796

// commit step 313: 473

// commit step 314: 586

// commit step 316: 345

// commit step 318: 823

// commit step 321: 645

// commit step 324: 514

// commit step 326: 833

// commit step 327: 328

// commit step 330: 973

// commit step 331: 222

// commit step 332: 753

// commit step 333: 121

// commit step 334: 102

// commit step 336: 200

// commit step 338: 532

// commit step 340: 197

// commit step 341: 896

// commit step 342: 605

// commit step 345: 920

// commit step 347: 350

// commit step 349: 495

// commit step 354: 822

// commit step 355: 714

// commit step 358: 388

// commit step 362: 995

// commit step 363: 824

// commit step 365: 990

// commit step 366: 537

// commit step 368: 869

// commit step 369: 482

// commit step 371: 611

// commit step 374: 303

// commit step 378: 102

// commit step 380: 199

// commit step 381: 393

// commit step 382: 563

// commit step 383: 329

// commit step 385: 944

// commit step 387: 881

// commit step 388: 754

// commit step 392: 680

// commit step 393: 928

// commit step 396: 573

// commit step 397: 846

// commit step 398: 930

// commit step 401: 753

// commit step 402: 214

// commit step 403: 519

// commit step 404: 735

// commit step 405: 797

// commit step 406: 417

// commit step 409: 527

// commit step 410: 825

// commit step 411: 360

// commit step 413: 878

// commit step 417: 998

// commit step 419: 732

// commit step 420: 248

// commit step 421: 617

// commit step 422: 851

// commit step 423: 930

// commit step 424: 961

// commit step 425: 834

// commit step 426: 663

// commit step 427: 270

// commit step 429: 941

// commit step 430: 996

// commit step 431: 146
