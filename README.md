# MDG tmu and python environment installation guide

## Quick Start

### Devcontainer
Devcontainer configuration is provided for VSCode.

- Remote SSH into MDG group server
- Clone THIS repo somewhere.
- Open repo as a folder (project) in VSCode.
- Make sure `.devcontainer/build` script is executable with the following command:
```bash
chmod +x ./.devcontainer/build
```
- Run the `.devcontainer/build` script to create `devcontainer.json` file for the devcontainer, with proper user and stuff.
```bash
./.devcontainer/build
```

- When prompted to open in devcontainer, select "Reopen in Container".
    - If not prompted, open the command palette (Ctrl+Shift+P), and select "Dev Containers: Reopen in Container".
- You should be dropped in `/workspace` folder inside the container, with all the files and environment installed.
- Any changes to files in the devcontainer will be reflected on the host machine, and vice versa.


## Details

### Environment Setup
Uses [Pixi](https://pixi.sh) to create and manage the environment. The environement is defined in the `pixi.yaml` file. 
To create the environment, run:

```bash
pixi install
```

### Installing additaional packages
If some package is missing from the environment, you can add it using `pixi add`.
If the package is available with conda:
``` bash
pixi add <package-name> 
```
If the package is only available with pip:
``` bash
pixi add --pypi <package-name>
```

### Activating the environment.
If you are using VsCode, the environement should get activated automatically.
To activate it manually:
``` bash
pixi shell
```



