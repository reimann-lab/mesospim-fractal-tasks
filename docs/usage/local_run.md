# Using Shell Commands

All tasks in this package can also be executed **outside of Fractal** using standard
shell commands. This is useful for:

- testing and debugging
- running the tasks locally
- integrating the tasks into custom scripts or pipelines
- development and verification

Each task exposes a CLI entry point installed with the package.

## General Pattern

Tasks can be executed as follow:

```bash
task-command --args1 <arg1> --args2 <arg2> ...
```

It is however recommended to use the JSON file to pass arguments to the task. A template exists for each task in the `examples` folder. The JSON file can be edited to only change the parameters of the task.

```bash
task-command --args-file params.json
```

Explanation about each parameter of each task can be found in the [Task Reference](add correct link) section of the documentation.