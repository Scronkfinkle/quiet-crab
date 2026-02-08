{
  description = "Quiet Crab Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; config.cudaSupport = true; };

    stdInputs = [
        # Rust
        pkgs.cargo
        pkgs.rustc
        pkgs.pkg-config
        pkgs.glib

        # Vulkan loader so wgpu can dlopen libvulkan.so.1 at runtime
        pkgs.vulkan-loader
    ];
    devInputs = [
        pkgs.rustfmt
        pkgs.clippy
        pkgs.rust-analyzer
    ];
  in
  {
    packages."x86_64-linux" = {
      # Full build with all features (default)
      default = pkgs.rustPlatform.buildRustPackage {
        pname = "quiet-crab";
        version = "0.1.0";
        src = ./.;
        buildType = "debug";
        cargoHash = "sha256-uhOSlUD0qVBfSEHZkOm9OTjzxYuUHAvvBRjAfmcYkCU=";

        buildInputs = stdInputs;

        nativeBuildInputs = stdInputs;

        # Point wgpu/Vulkan at the system NVIDIA ICD (NixOS places it here)
        VK_ICD_FILENAMES = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json";
        # Needed to find the Vulkan loader (libvulkan.so.1) and NVIDIA driver libs at runtime
        LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib:${pkgs.addDriverRunpath.driverLink}/lib";
      };
    };
    devShells."x86_64-linux".default = pkgs.mkShell {
       buildInputs = stdInputs ++ devInputs;

       # Rust stdlib for language servers
       RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
       # Point wgpu/Vulkan at the system NVIDIA ICD (NixOS places it here)
       VK_ICD_FILENAMES = "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json";
       # Needed to find the Vulkan loader (libvulkan.so.1) and NVIDIA driver libs at runtime
       LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib:${pkgs.addDriverRunpath.driverLink}/lib";
       };
  };

}
