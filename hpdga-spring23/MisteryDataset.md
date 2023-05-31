# How to download

Since this dataset is way bigger than the others, expect the loading time of the graph to be long.

Remember to change the GCN parameters to the ones for the reddit dataset.

The accuracy is low for the given number of epochs. To avoid a very long training process, your solution will be considered valid (for this dataset) if it reduces the needed training time while maintaining the same level of accuracy as the sequential implementation for the given number of epochs.
You are free to experiment with the model hyperparameters to increase the accuracy.

Install DriveDownloader to download Onedrive files:

```sh
!pip install DriveDownloader
```

Then download the files for the reddit dataset

```sh
!ddl https://polimi365-my.sharepoint.com/:u:/g/personal/10580652_polimi_it/EZ2sYEbUVJxCtrIAFKX8q7YBXLxaZ3VfHgUU655WYxn6Tw?e=CADLjR -o ./data/reddit.graph

!ddl https://polimi365-my.sharepoint.com/:u:/g/personal/10580652_polimi_it/EaQ4s6QZC9RErRSTmROshqwBab5JZUBJrY7AmQXjnkZqtQ?e=K3QCZf -o ./data/reddit.split

!ddl https://polimi365-my.sharepoint.com/:u:/g/personal/10580652_polimi_it/EfuCLAlxlZhIjXuZgDLeknEBgdN2507Q4q79PLpKK9OD5A?e=afgtFu -o ./data/reddit.svmlight
```

You can then launch your app on the new dataset by calling it with the "reddit" name.