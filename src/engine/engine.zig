//! This file provides the autograd engine functionality for kiwigrad

/// The type of expression
pub const ExprType = enum {
    nop,
    unary,
    binary,
};

pub const UnaryType = enum {
    tanh,
    exp,
    relu,
    softmax,

    pub fn toString(self: UnaryType) []const u8 {
        return switch (self) {
            .tanh => "tanh",
            .exp => "^",
            .relu => "ReLU",
            .softmax => "Softmax",
        };
    }
};

/// Unary operation structure
pub fn UnaryOp(comptime ValueType: type) type {
    return struct {
        /// The unary operation that produced the value
        op: UnaryType,
        /// The backpropagation function
        backprop_fn: *const fn (*ValueType) void,
        /// The children used to compute the value
        prev: [1]*ValueType,
    };
}

pub const BinaryType = enum {
    add,
    sub,
    mul,
    div,

    pub fn toString(self: BinaryType) []const u8 {
        return switch (self) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
        };
    }
};

/// Binary operation structure
pub fn BinaryOp(comptime ValueType: type) type {
    return struct {
        /// The binary operation that produced the value
        op: BinaryType,
        /// The backpropagation function
        backprop_fn: *const fn (*ValueType) void,
        /// The children used to compute the value
        prev: [2]*ValueType,
    };
}

pub const Scalar = @import("scalar.zig").Scalar;
pub const Array = @import("tensor.zig").Array;
pub const Tensor = @import("tensor.zig").Tensor;
