{
  description = "pgp-vanity";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # Only the .#cuda variant pulls in unfree packages; the default
          # ROCm package stays free.
          config.allowUnfreePredicate =
            pkg:
            builtins.elem (nixpkgs.lib.getName pkg) [
              "cuda_nvrtc"
            ];
        };
        mkPackage =
          wrapperArgs:
          pkgs.rustPlatform.buildRustPackage {
            pname = "pgp-vanity";
            version = "0.1.0";

            src = ./.;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            nativeBuildInputs = with pkgs; [
              pkg-config
              rustPlatform.bindgenHook
              makeWrapper
            ];

            buildInputs = with pkgs; [ ];

            postInstall = ''
              wrapProgram $out/bin/pgp-vanity ${wrapperArgs}
            '';
          };
        hipPackage = mkPackage ''
          --set ROCM_PATH ${pkgs.rocmPackages.clr} \
          --prefix LD_LIBRARY_PATH : "${
            pkgs.lib.makeLibraryPath [
              pkgs.rocmPackages.clr
              pkgs.rocmPackages.rocm-runtime
            ]
          }"
        '';
        # libcuda.so.1 must come from the host NVIDIA driver. Systems that
        # expose the driver at /run/opengl-driver/lib need that entry; on
        # other distros the dynamic loader finds libcuda on its own, and the
        # extra path is harmless. NVRTC comes from nixpkgs.
        cudaPackage = mkPackage ''
          --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib:${
            pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages.cuda_nvrtc
            ]
          }"
        '';
      in
      {
        packages.default = hipPackage;
        packages.hip = hipPackage;
        packages.cuda = cudaPackage;
        checks.default = hipPackage;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer
            pkg-config
            rocmPackages.clr
            rocmPackages.rocminfo
            rocmPackages.rocm-runtime
          ];

          ROCM_PATH = pkgs.rocmPackages.clr;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.rocmPackages.clr pkgs.rocmPackages.rocm-runtime ];
        };
      }
    );
}
