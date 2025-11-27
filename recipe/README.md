# About Deprecating Recipes in the Main Repository

> [!NOTE]
> Most previous recipes have been migrated to a dedicated repository [`verl-recipe`](https://github.com/verl-project/verl-recipe) due to lack of maintenance from the original contributors.
>
> For historical reasons, here are still some recipe projects because
>
> - they are expected to be decomposed into independent modules and integrated into verl in the future,
> - and there are some awesome contributors from the community actively maintaining and developing them.

`verl` is designed to be a modular library for flexible and efficient post-training.

Developers are expected to `import verl` as a package with necessary extensions to build specific post-training pipeline.

If you find your end-to-end recipe might be useful for others, welcome to create your own repository and [register it](../docs/start/awesome_projects.rst).

If you find your extension can be modular and useful for others, welcome to [contribute to `verl](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md).
