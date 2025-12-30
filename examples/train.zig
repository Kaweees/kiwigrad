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
    const model = MLPType.new(sizes.len - 1, sizes[0..]);

    // dataset
    const X = [_][3]f64{
        .{ 2.0, 3.0, -1.0 },
        .{ 3.0, -1.0, 0.5 },
        .{ 0.5, 1.0, 1.0 },
        .{ 1.0, 1.0, -1.0 },
    };
    const y = [_]f64{ 1.0, -1.0, -1.0, 1.0 };

    const lr = 1e-2;
    const epochs: usize = 100;

    // training loop
    for (0..epochs) |epoch| {
        // Zero out the gradients
        model.zero_grad();

        // Accumulate loss across all samples (like the reference implementation)
        var loss: ?*ValueType = null;
        var first = true;

        var i: usize = 0;
        while (i < X.len) : (i += 1) {
            var inputs: [X[i].len]*ValueType = undefined;
            for (&inputs, 0..) |*input, j| {
                input.* = ValueType.new(X[i][j]);
            }

            const z = model.forward(&inputs);
            const ypred = z[0];

            const ygt = ValueType.new(y[i]);
            const diff = ypred.sub(ygt);
            const sq = diff.mul(diff);

            if (first) {
                loss = sq;
                first = false;
            } else {
                loss = loss.?.add(sq);
            }
        }

        const total_loss = loss.?.data;

        // Single backward pass on accumulated loss
        loss.?.backwardPass(alloc);

        // Update parameters with SGD
        model.update_parameters(lr);

        std.debug.print("Epoch={d:4} loss={d:.6}\n", .{ epoch, total_loss });
    }
}
