{
  description = "Zotero MCP server and CLI";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          py = pkgs.python3Packages;

          # nixpkgs has an older pyzotero; pyproject.toml needs >=1.14.0. 1.15+ moved to httpx2.
          pyzotero = py.buildPythonPackage rec {
            pname = "pyzotero";
            version = "1.14.0";
            pyproject = true;
            src = py.fetchPypi {
              inherit pname version;
              hash = "sha256-fNszx6WSuGUCZg7R71GIavcqI/XUZ7cRKUUiBE9B14s=";
            };
            # upper bound on the build backend only; nixpkgs moves ahead of it
            postPatch = ''
              substituteInPlace pyproject.toml --replace-fail ',<0.12.0' ""
            '';
            build-system = [ py.uv-build ];
            dependencies = with py; [
              feedparser
              bibtexparser
              httpx
              whenever
            ];
            pythonImportsCheck = [ "pyzotero" ];
          };

          # Not in nixpkgs: Rust extension, built from the sdist with maturin.
          pdf-inspector = py.buildPythonPackage rec {
            pname = "pdf-inspector";
            version = "0.2.6";
            pyproject = true;
            src = py.fetchPypi {
              pname = "pdf_inspector";
              inherit version;
              hash = "sha256-W7OH85v3qTsCtJGItnC5eY+MzH5Y9o7uWIOlEqoFzrI=";
            };
            cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
              inherit pname version src;
              hash = "sha256-/PTqpmL2JdnK/Ejo3IAK/DqTSVrA9zTmFnmRPoc4tLc=";
            };
            nativeBuildInputs = with pkgs.rustPlatform; [
              cargoSetupHook
              maturinBuildHook
            ];
            pythonImportsCheck = [ "pdf_inspector" ];
          };

          zotero-mcp = py.buildPythonApplication {
            pname = "zotero-mcp-server";
            version = builtins.head (
              builtins.match ".*__version__ = \"([^\"]+)\".*" (builtins.readFile ./src/zotero_mcp/_version.py)
            );
            pyproject = true;
            src = self;
            build-system = [ py.hatchling ];
            dependencies = [
              pyzotero
              pdf-inspector
            ]
            ++ (with py; [
              python-dotenv
              markdownify
              markdown-it-py
              pydantic
              requests
              fastmcp
              httpx
              unidecode
              bibtexparser
            ]);
            pythonImportsCheck = [ "zotero_mcp" ];
            meta = {
              description = "Zotero MCP server and CLI";
              homepage = "https://github.com/54yyyu/zotero-mcp";
              mainProgram = "zotero-mcp";
            };
          };
        in
        {
          inherit zotero-mcp;
          default = zotero-mcp;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];
          };
        }
      );
    };
}
