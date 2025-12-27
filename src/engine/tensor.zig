//! This file provides the autograd engine functionality for kiwigrad

const std = @import("std");
const engine = @import("engine.zig");

/// Represents a multi-dimensional array
pub fn Array(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The data
        data: []T,
        /// The shape of the array
        shape: []usize,
        /// The stride of the array
        stride: []usize,
        /// The number of dimensions of the array
        dims: usize,
        /// The number of elements in the array
        size: usize,

        var arena: std.heap.ArenaAllocator = undefined;

        pub fn init(alloc: std.mem.Allocator) void {
            arena = std.heap.ArenaAllocator.init(alloc);
        }

        /// Cleanup allocated memory
        pub fn deinit() void {
            arena.deinit();
        }

        /// Create a new Array
        pub fn new(data: []T, shape: []usize, stride: []usize, dims: usize, size: usize) *Self {
            const a = arena.allocator().create(Self) catch unreachable;
            a.* = Self{
                .data = data,
                .shape = shape,
                .stride = stride,
                .dims = dims,
                .size = size,
            };
            return a;
        }

        /// Find the element at the given coordinates
        pub inline fn at(self: *Self, coords: []const usize) *T {
            return self.data[self.index(coords)];
        }

        /// Find the index of the element at the given coordinates
        pub inline fn index(self: *Self, coords: []const usize) usize {
            if (coords.len != self.dims) {
                std.debug.panic("Input size mismatch: {d} != {d}", .{ coords.len, self.dims });
            }

            var idx = 0;
            for (coords, 0..) |coord, i| {
                idx += coord * self.stride[i];
            }
            return idx;
        }

        /// Set the element at the given coordinates
        pub inline fn set(self: *Self, coords: []const usize, value: T) void {
            self.data[self.index(coords)] = value;
        }
    };
}

/// Represents an auto-differentiable Tensor value
pub fn Tensor(comptime T: type) type {
    const ArrayType = Array(T);
    // Check that T is a valid type
    switch (@typeInfo(T)) {
        .int, .comptime_int, .float, .comptime_float => {},
        else => @compileError("Expected @int or @float type, got: " ++ @typeName(T)),
    }

    return struct {
        const Self = @This();
        const BackpropFn = *const fn (self: *Self) void;

        const Expr = union(engine.ExprType) {
            nop: void,
            unary: engine.UnaryOp(Self),
            binary: engine.BinaryOp(Self),
        };

        /// The data
        data: []Array(T),
        /// The gradient
        grad: []Array(T),
        /// The expression that produced the value
        expr: Expr,

        /// The arena allocator
        var arena: std.heap.ArenaAllocator = undefined;

        /// Initialize the arena allocator
        pub fn init(alloc: std.mem.Allocator) void {
            arena = std.heap.ArenaAllocator.init(alloc);
        }

        /// Deinitialize the arena allocator
        pub fn deinit() void {
            arena.deinit();
        }

        /// Create a new Tensor value from array data
        pub fn new(data: []const T) *Self {
            const t = arena.allocator().create(Self) catch unreachable;

            // Copy the input data to our own allocation
            const tensor_data = arena.allocator().alloc(T, data.len) catch unreachable;
            @memcpy(tensor_data, data);

            // Create shape, stride for 1D tensor
            const shape = arena.allocator().alloc(usize, 1) catch unreachable;
            const stride = arena.allocator().alloc(usize, 1) catch unreachable;
            shape[0] = data.len;
            stride[0] = 1;

            // Create the data array
            const data_array = ArrayType.new(tensor_data, shape, stride, 1, data.len);

            // Create gradient array (initialized to zeros)
            const grad_data = arena.allocator().alloc(T, data.len) catch unreachable;
            @memset(grad_data, 0);
            const grad_shape = arena.allocator().alloc(usize, 1) catch unreachable;
            const grad_stride = arena.allocator().alloc(usize, 1) catch unreachable;
            grad_shape[0] = data.len;
            grad_stride[0] = 1;
            const grad_array = ArrayType.new(grad_data, grad_shape, grad_stride, 1, data.len);

            // Create arrays to hold the Array values (not pointers)
            const data_arrays = arena.allocator().alloc(ArrayType, 1) catch unreachable;
            const grad_arrays = arena.allocator().alloc(ArrayType, 1) catch unreachable;
            data_arrays[0] = data_array.*;
            grad_arrays[0] = grad_array.*;

            t.* = Self{ .data = data_arrays, .grad = grad_arrays, .expr = .{ .nop = {} } };

            return t;
        }

        // /// Add two Tensors
        // pub inline fn add(self: *Self, other: *Self) *Self {
        //   return binary(self.data + other.data, .add, add_back, self, other);
        // }
    };
}
