from invoke import Collection, Context, task


@task()
def lint(c: Context) -> None:
    """
    Formats code
    """
    c.run("black --exclude=venv .")


ns = Collection()
ns.add_task(lint)
