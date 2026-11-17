<#
.SYNOPSIS
Publish a clean source export as a new public KHOSTspec repository.

.DESCRIPTION
Run this script from the reviewed source-only export after signing in with
GitHub CLI. It creates the personal-account repository, uploads its initial
main branch, creates develop for future work, and enables GitHub Pages with
main as the permitted deployment branch. The main push triggers the
documentation workflow. It refuses an existing repository or local .git entry.
It does not reuse the original working repository's history.

GitHub tokens are handled by GitHub CLI and are never requested or printed.
The initial commit uses the account's GitHub noreply address. Git identity and
credential settings are supplied per command rather than changed globally.

.EXAMPLE
gh auth login --hostname github.com --web --git-protocol https --scopes repo,workflow
powershell -File tools/publish_github.ps1

.NOTES
This script publishes source code publicly. If a step fails after creation,
inspect the new local and remote repositories before continuing manually.
No rollback, deletion, force push, or automatic overwrite is performed.
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-CheckedCommand {
    <#
    .SYNOPSIS
    Run one native command and stop on failure without exposing its response.
    .DESCRIPTION
    Capture native output for selected metadata checks. AllowFailure is used
    only when the caller must distinguish an HTTP 404 from other failures.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$Step,
        [switch]$AllowFailure
    )
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $outputLines = @(& $Executable @ArgumentList 2>&1 | ForEach-Object { [string]$_ })
        $commandExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if (($commandExitCode -ne 0) -and (-not $AllowFailure)) {
        throw "$Step failed (exit code $commandExitCode). Inspect the named command locally before continuing."
    }
    return [pscustomobject]@{ ExitCode = $commandExitCode; Output = $outputLines }
}

function Get-ApiHttpStatus {
    <#
    .SYNOPSIS
    Read a GitHub API HTTP status without treating network failure as absence.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$GhExecutable,
        [Parameter(Mandatory = $true)][string]$Endpoint,
        [Parameter(Mandatory = $true)][string[]]$Headers
    )
    $response = Invoke-CheckedCommand -Executable $GhExecutable -ArgumentList (
        @('api', '--hostname', 'github.com', '--method', 'GET', '--include', '--silent') +
        $Headers + @($Endpoint)
    ) -Step 'Read repository setup status' -AllowFailure
    $statusMatch = [regex]::Match(($response.Output -join "`n"), '(?m)^HTTP/\S+\s+([0-9]{3})\b')
    if (-not $statusMatch.Success) {
        throw 'GitHub returned no HTTP status. Check the connection and gh authentication; no absence is assumed.'
    }
    return [int]$statusMatch.Groups[1].Value
}

function Test-PublicSourcePath {
    <#
    .SYNOPSIS
    Accept only the maintained source paths intended for the initial upload.
    #>
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $rootFiles = @(
        '.gitignore', 'README.md', 'requirements.txt', 'requirements-dev.txt',
        'requirements-docs.txt', 'spec_reduction_parallel.py', 'speclibparallel.py',
        'config.example.yaml', 'NeAr.csv', 'tests/test_entrypoint.py',
        'docs/conf.py', 'docs/build.py', 'docs/_static/custom.css',
        '.github/workflows/documentation.yml', 'tools/publish_github.ps1'
    )
    return (($RelativePath -cin $rootFiles) -or
        ($RelativePath -cmatch '^docs/[A-Za-z0-9_-]+\.rst$') -or
        ($RelativePath -cmatch '^docs/api/[A-Za-z0-9_-]+\.rst$'))
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repositoryName = 'KHOSTspec'
$branchName = 'main'
$developmentBranchName = 'develop'
$previousGhHost = [Environment]::GetEnvironmentVariable('GH_HOST', 'Process')
$localRepositoryCreated = $false
$remoteRepositoryCreated = $false

try {
    if (Test-Path -LiteralPath (Join-Path $projectRoot '.git')) {
        throw 'A .git entry already exists here. Use a fresh reviewed source export; the existing history will not be reused.'
    }
    foreach ($routingVariable in @('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR')) {
        if ([Environment]::GetEnvironmentVariable($routingVariable, 'Process')) {
            throw "$routingVariable redirects Git operations. Run this helper from a shell without Git routing overrides."
        }
    }
    foreach ($requiredPath in @('.gitignore', 'README.md', 'speclibparallel.py', '.github/workflows/documentation.yml')) {
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $requiredPath) -PathType Leaf)) {
            throw "The reviewed source export is incomplete: $requiredPath is missing."
        }
    }
    $ghCommand = Get-Command gh -CommandType Application -ErrorAction SilentlyContinue
    $gitCommand = Get-Command git -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $ghCommand) {
        throw 'GitHub CLI is required. Install it from https://cli.github.com/ and run gh auth login --hostname github.com --web --git-protocol https --scopes repo,workflow.'
    }
    if ($null -eq $gitCommand) {
        throw 'Git is required. Install Git, reopen PowerShell, and run this script from a fresh source export.'
    }
    $ghExecutable = $ghCommand.Source
    $gitExecutable = $gitCommand.Source
    $env:GH_HOST = 'github.com'
    $apiHeaders = @('-H', 'Accept: application/vnd.github+json', '-H', 'X-GitHub-Api-Version: 2026-03-10')

    Write-Host 'Checking the authenticated GitHub account.'
    $null = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList @(
        'auth', 'status', '--hostname', 'github.com'
    ) -Step 'GitHub authentication check; use gh auth login --web if needed'
    $identityResponse = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList (
        @('api', '--hostname', 'github.com') + $apiHeaders + @('user', '--jq', '{login: .login, id: .id}')
    ) -Step 'Read the authenticated account login and numeric ID'
    $identity = ($identityResponse.Output -join "`n") | ConvertFrom-Json
    $owner = [string]$identity.login
    $accountId = [long]$identity.id
    if (($owner -notmatch '^[A-Za-z0-9][A-Za-z0-9-]{0,38}$') -or ($accountId -le 0)) {
        throw 'GitHub did not return a valid personal-account identity.'
    }
    $repository = "$owner/$repositoryName"
    $repositoryEndpoint = "repos/$repository"
    $repositoryStatus = Get-ApiHttpStatus -GhExecutable $ghExecutable -Endpoint $repositoryEndpoint -Headers $apiHeaders
    if ($repositoryStatus -eq 200) {
        throw "$repository already exists. This one-time helper will not overwrite or reuse it."
    }
    if ($repositoryStatus -ne 404) {
        throw "Cannot confirm that $repository is available (HTTP $repositoryStatus)."
    }

    Write-Host "Preparing a fresh $branchName branch for $repository."
    $gitArguments = @('-C', $projectRoot)
    $null = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('init', "--initial-branch=$branchName")
    ) -Step 'Initialize the source-only Git repository'
    $localRepositoryCreated = $true
    $rootResponse = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('rev-parse', '--show-toplevel')
    ) -Step 'Verify the new Git working directory'
    if ([System.IO.Path]::GetFullPath(($rootResponse.Output -join '').Trim()) -ne $projectRoot) {
        throw 'Git selected a different working directory. No source files will be staged or uploaded.'
    }
    $null = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('add', '--', '.')
    ) -Step 'Stage the maintained source allowlist'
    $stagedResponse = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('ls-files')
    ) -Step 'Check staged source paths'
    if ($stagedResponse.Output.Count -eq 0) {
        throw 'No maintained source files were staged.'
    }
    foreach ($relativePath in $stagedResponse.Output) {
        if (-not (Test-PublicSourcePath -RelativePath $relativePath)) {
            throw "Unexpected staged path: $relativePath. Review the local export before uploading."
        }
    }
    $stagedModes = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('ls-files', '--stage')
    ) -Step 'Check staged file types'
    if (@($stagedModes.Output | Where-Object { $_ -match '^120000 ' }).Count -gt 0) {
        throw 'The staged export contains a symbolic link. Resolve it before uploading.'
    }
    $noreplyEmail = "$accountId+$owner@users.noreply.github.com"
    $null = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('-c', "user.name=$owner", '-c', "user.email=$noreplyEmail",
            'commit', '-m', 'Initial KHOSTspec source and documentation')
    ) -Step 'Create the initial commit with a GitHub noreply address'
    $commitIdentity = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('show', '-s', '--format=%an|%ae|%cn|%ce', 'HEAD')
    ) -Step 'Verify the initial commit identity before uploading'
    if (($commitIdentity.Output -join '').Trim() -cne "$owner|$noreplyEmail|$owner|$noreplyEmail") {
        throw 'Commit identity differs from the intended GitHub noreply identity. Inspect Git author and committer environment overrides before uploading.'
    }

    Write-Host "Creating the public repository $repository."
    $null = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList @(
        'repo', 'create', $repository, '--public', '--source', $projectRoot,
        '--remote', 'origin', '--description', 'KHAO optical spectral reduction pipeline and documentation'
    ) -Step 'Create the new public GitHub repository'
    $remoteRepositoryCreated = $true
    $null = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
            'push', '--set-upstream', 'origin', $branchName)
    ) -Step 'Push the initial main branch'
    $null = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList @(
        'repo', 'edit', $repository, '--default-branch', $branchName
    ) -Step 'Set main as the default branch'
    $null = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('branch', $developmentBranchName, 'HEAD')
    ) -Step 'Create develop for future development'
    $null = Invoke-CheckedCommand -Executable $gitExecutable -ArgumentList (
        $gitArguments + @('-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
            'push', '--set-upstream', 'origin', $developmentBranchName)
    ) -Step 'Push the initial develop branch without publishing documentation'

    Write-Host 'Enabling GitHub Pages and the main deployment environment.'
    $pagesEndpoint = "$repositoryEndpoint/pages"
    $pagesStatus = Get-ApiHttpStatus -GhExecutable $ghExecutable -Endpoint $pagesEndpoint -Headers $apiHeaders
    if ($pagesStatus -eq 404) {
        $null = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList (
            @('api', '--hostname', 'github.com', '--method', 'POST') + $apiHeaders +
            @($pagesEndpoint, '-f', 'build_type=workflow', '--silent')
        ) -Step 'Enable GitHub Actions as the Pages source'
    }
    elseif ($pagesStatus -eq 200) {
        $pagesMode = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList (
            @('api', '--hostname', 'github.com') + $apiHeaders + @($pagesEndpoint, '--jq', '.build_type')
        ) -Step 'Verify the Pages source configured during the first push'
        if (($pagesMode.Output -join '').Trim() -ne 'workflow') {
            throw 'Pages already has a different source. Inspect repository settings before continuing.'
        }
    }
    else {
        throw "Cannot inspect the new Pages configuration (HTTP $pagesStatus)."
    }
    $environmentEndpoint = "$repositoryEndpoint/environments/github-pages"
    $null = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList (
        @('api', '--hostname', 'github.com', '--method', 'PUT') + $apiHeaders +
        @($environmentEndpoint, '-F', 'deployment_branch_policy[protected_branches]=false',
            '-F', 'deployment_branch_policy[custom_branch_policies]=true', '--silent')
    ) -Step 'Configure the Pages deployment environment'
    $policiesEndpoint = "$environmentEndpoint/deployment-branch-policies"
    $existingPolicies = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList (
        @('api', '--hostname', 'github.com') + $apiHeaders +
        @($policiesEndpoint, '--jq', '.branch_policies[].name')
    ) -Step 'Read existing deployment branch names'
    if ($branchName -cnotin $existingPolicies.Output) {
        $null = Invoke-CheckedCommand -Executable $ghExecutable -ArgumentList (
            @('api', '--hostname', 'github.com', '--method', 'POST') + $apiHeaders +
            @($policiesEndpoint, '-f', "name=$branchName", '-f', 'type=branch', '--silent')
        ) -Step 'Allow main deployments to GitHub Pages'
    }

    Write-Host "Repository: https://github.com/$repository"
    Write-Host "Actions: https://github.com/$repository/actions/workflows/documentation.yml"
    Write-Host "Pages: https://$owner.github.io/$repositoryName/"
    Write-Host 'The initial main push triggers the documentation workflow. Check Actions for the deployment result.'
    Write-Host 'If deployment ran before Pages was enabled, rerun the failed jobs from Actions after this setup finishes.'
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    if ($localRepositoryCreated -or $remoteRepositoryCreated) {
        Write-Host 'Setup stopped after partial progress. Inspect the local .git and any new GitHub repository before continuing manually.'
    }
    exit 1
}
finally {
    if ($null -eq $previousGhHost) {
        Remove-Item Env:\GH_HOST -ErrorAction SilentlyContinue
    }
    else {
        $env:GH_HOST = $previousGhHost
    }
}
