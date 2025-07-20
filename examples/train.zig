//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.

const std = @import("std");
const kiwigrad = @import("kiwigrad");
const zbench = @import("zbench");

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    // Initialize the required components
    const ValueType = kiwigrad.engine.Scalar(f64);
    const NeuronType = kiwigrad.nn.Neuron(f64);
    const LayerType = kiwigrad.nn.Layer(f64);
    const MLPType = kiwigrad.nn.MLP(f64);
    const ArrayType = kiwigrad.engine.Array(f64);
    const TensorType = kiwigrad.engine.Tensor(f64);

    // Initialize allocators and components
    ValueType.init(alloc);
    NeuronType.init(alloc);
    LayerType.init(alloc);
    MLPType.init(alloc);
    ArrayType.init(alloc);
    TensorType.init(alloc);
    defer {
        ValueType.deinit();
        NeuronType.deinit();
        LayerType.deinit();
        MLPType.deinit();
        ArrayType.deinit();
        TensorType.deinit();
    }

    var sizes = [_]usize{ 3, 2, 1 };

    // Initialize the neuron
    const mlp = MLPType.new(sizes.len - 1, sizes[0..]);

    const inputs = [_][3]*ValueType{
        [_]*ValueType{ ValueType.new(2), ValueType.new(3), ValueType.new(-1) },
        [_]*ValueType{ ValueType.new(3), ValueType.new(-1), ValueType.new(0.5) },
        [_]*ValueType{ ValueType.new(0.5), ValueType.new(1), ValueType.new(1) },
        [_]*ValueType{ ValueType.new(1), ValueType.new(2), ValueType.new(3) },
    };

    mlp.draw_graph("assets/img/mlp");

    for (inputs) |in| {
        // Forward pass through the layer
        const output = mlp.forward(@constCast(&in));
        std.debug.print("{d:7.4} ", .{output[0].data});
        for (output) |o| {
            _ = o.draw_graph("assets/img/perceptron");
        }
    }

    const t1 = TensorType.new(&[_]f64{ 1, 2, 3, 4 });
    std.debug.print("t1: {d:.4}\n", .{t1.data[0].data});

    // // outputs now contains 2 ValueType pointers (one for each neuron)
    // print("Layer output: {d:.4}\n", .{output.data});

    // print("output.data: {d:.4}\n", .{output.data});
    // print("output.grad: {d:.4}\n", .{output.grad});

    // output.backwardPass(alloc);

    // print("output.data: {d:.4}\n", .{output.data});
    // print("output.grad: {d:.4}\n", .{output.grad});
}
