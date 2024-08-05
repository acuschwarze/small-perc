param (
    [int]$MaxParallelJobs = 20
)

$env:path += ';C:\Users\f00689q\AppData\Local\miniforge3\envs\mathai\'

# Define the input parameters
$inputParameters = 1..100 | ForEach-Object {$_ * 0.01 }

# Array to hold job objects
$jobs = @()
# Function to check the number of running jobs
function Get-RunningJobCount {
    return (Get-Job -State Running).Count
}
# Start a job for each input parameter
foreach ($param in $inputParameters) {
    # Wait if the number of running jobs is equal to or greater than the maximum allowed
    $jobcount = (Get-Job | Where-Object { $_.State -eq 'Running' }).Count # needs to be called again to work
    while ($jobcount -ge $MaxParallelJobs) {
        Start-Sleep -Seconds 1;
        $jobcount = (Get-Job | Where-Object { $_.State -eq 'Running' }).Count
    }
    # Start the job
    Write-Host $param
    $jobs += Start-Job -ScriptBlock {
        param ($param)
        # Path to your script
        & python "C:\Users\f00689q\My Drive\jupyter\small-perc\heatmap_infinite.py" $param
    } -ArgumentList $param
}
# Wait for all jobs to complete
$jobs | ForEach-Object { $_ | Wait-Job }
# Get job results
$jobs | ForEach-Object {
    $result = Receive-Job -Job $_
    Write-Host $result
    Remove-Job -Job $_
}