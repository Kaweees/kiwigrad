//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.

const std = @import("std");
const kiwigrad = @import("kiwigrad");

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    // Initialize the required components
    const ValueType = kiwigrad.engine.Scalar(f64);
    const NeuronType = kiwigrad.nn.Neuron(f64);
    const LayerType = kiwigrad.nn.Layer(f64);
    const MLPType = kiwigrad.nn.MLP(f64);

    // Initialize allocators and components
    ValueType.init(alloc);
    NeuronType.init(alloc);
    LayerType.init(alloc);
    MLPType.init(alloc);
    defer {
        ValueType.deinit();
        NeuronType.deinit();
        LayerType.deinit();
        MLPType.deinit();
    }

    var sizes = [_]usize{ 3, 2, 1 };

    // Initialize the neural network
    const mlp = MLPType.new(sizes.len - 1, sizes[0..]);

    const inputs = [_][3]*ValueType{
        [_]*ValueType{ ValueType.new(2), ValueType.new(3), ValueType.new(-1) },
        [_]*ValueType{ ValueType.new(3), ValueType.new(-1), ValueType.new(0.5) },
        [_]*ValueType{ ValueType.new(0.5), ValueType.new(1), ValueType.new(1) },
        [_]*ValueType{ ValueType.new(1), ValueType.new(2), ValueType.new(3) },
    };

    mlp.draw_graph("assets/img/mlp");

    var output: []*ValueType = undefined;
    for (inputs) |in| {
        // Forward pass through the layer
        output = mlp.forward(@constCast(&in));
        std.debug.print("Layer output: {d:7.4}\n", .{output[0].data});
        for (output) |o| {
            _ = o.draw_graph("assets/img/perceptron");
        }
    }

    const t1 = TensorType.new(&[_]f64{ 1, 2, 3, 4 });
    std.debug.print("t1: {d:.4}\n", .{t1.data[0].data});

    // outputs now contains 1 ValueType pointer (final layer has 1 neuron)
    const final_output = output[0];
    std.debug.print("Layer output: {d:.4}\n", .{final_output.data});

    std.debug.print("output.data: {d:.4}\n", .{final_output.data});
    std.debug.print("output.grad: {d:.4}\n", .{final_output.grad});

    final_output.backwardPass(alloc);

    std.debug.print("output.data: {d:.4}\n", .{final_output.data});
    std.debug.print("output.grad: {d:.4}\n", .{final_output.grad});
}
