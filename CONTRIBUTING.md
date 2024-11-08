## Guidelines

Welcome to the RIPPL development community! First of all, we welcome everyone who wants to contribute!

If you're only interested in playing around with RIPPL or testing a few new functions based on RIPPL, you can follow the guidelines described in `README.md`. If you want your ideas and code to be used by other researchers around the world to facilitate their research, we would like you to follow some general guidelines when contributing. This will ensure the robustness of the software. Some general rules are:

- Docstring: All public functions should have a docstring in `numpy` format;
- Syntax: All codes should be checked by the `flake8` linter;
- Unittest: EVERY FUNCTION should have uniitest implemented;
- Continuous Integration: All pull requests should first pass the CI test (when the CI integration is ready);

## Git Workflow

We will follow the git workflow guide from https://nvie.com/posts/a-successful-git-branching-model/. The general guidelines are:

1. `main` branch is **protected** and will only be updated when there's a major stable release (i.e., v1.2.0 -> v1.3.0);
2. `develop` branch is **protected**, and can be treated as a "beta" version. The "beta" version will be tested thoroughly before merging to `main` branch. 
3. All development work should be `checkout` from `develop` branch. When your work is done, submit a pull request to merge your work back to `develop` branch. 
4. All development work MUST have a corresponding ticket in Jira: https://doris.atlassian.net/jira for keeping track of your work. UNDOCUMENTED branches will NOT be accepted for pull request, and might be deleted by the administrator!

## Start your Development

For setting up the development environment, please refer to `README.md`. 

For following the git guideline, you can do the following:

```bash
git checkout develop  # start from develop branch
git checkout -b DORIS-XX_new_feature_here  # checkout new branch, new branch name must correspond to a jira ticket number
git push -u origin DORIS-XX_new_feature_here  # push to upstream
```

When you're done with implementing this new feature, you can submit a **pull request** on bitbucket. We will then review the pull request. 
