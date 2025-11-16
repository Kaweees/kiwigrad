{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    zig_0_14 # Zig compiler 0.14.1
    nixfmt # Nix formatter
    just # Just runner
    graphviz # Graphviz
  ];

  # Shell hook to set up environment
  shellHook = "";
}
