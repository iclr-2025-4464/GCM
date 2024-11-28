# Generative Cost Model

This is the project of ***Reformulating Strict Monotonic Probabilities with a Generative Cost Model***.

## Requirements
`python>=3.8.19`

`ucimlrepo>=0.0.7`

`tensorflow>=2.13.0`

`numpy>=1.24.3`

`pandas>=2.0.3`

## Quick Start
`python run.py {dataset_id} {random_seed}`

Here `dataset_id` should be in {0,1,2} and `random_seed` should be an integer.

The mapping of `dataset_id` to the actual public dataset is:

`
{
    0: Adult, 1: Diabetes, 2: BlogFeedback
}
`

## Datasets

The Adult dataset is downloaded from <https://archive.ics.uci.edu/dataset/2/adult>.

The Diabetes dataset is downloaded from <https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators>.

The BlogFeedback dataset is downloaded from <https://archive.ics.uci.edu/dataset/304/blogfeedback>.

