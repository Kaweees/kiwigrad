# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias b := build
alias r := run
alias t := test
alias c := clean
alias f := format
alias d := docs

# Default command when 'just' is run without arguments
default:
  @just --list -u

# Build the project
build *args="":
  @echo "Building..."
  @zig build --fetch --summary all {{args}}

# Run a package
run *args="":
  @echo "Running..."
  @zig build run -Doptimize=ReleaseFast {{args}}
  @dot -Tpng assets/img/mlp.dot -o assets/img/mlp.png
  @dot -Tpng assets/img/perceptron.dot -o assets/img/perceptron.png

# Test the project
test *args="":
  @echo "Testing..."
  @zig build test --summary all {{args}}

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @find . -type d -name ".zig-cache" -exec rm -rf {} +
  @find . -type d -name "zig-out" -exec rm -rf {} +

# Format the project
format:
  @echo "Formatting..."
  @zig fmt .
  @find . -name "*.nix" -type f -exec nixfmt {} \;

# Generate documentation
docs:
  @echo "Generating documentation..."
  @zig build docs
