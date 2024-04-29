## Set up the environment
```
pip install -r requirements.txt
```
Generate softlink for the dataset.
```
ln -s source_path otb_mini
```

## RUN MOSSE TEST

You need to download the otb_mini dataset [here](https://drive.google.com/drive/folders/1XzY7BJl7cElJekbq4eWk72EBxKVwp-y6?usp=sharing).


```
$ python test_mosse.py
```
NOTE:
1. There are two MOSSE tracker classes which are `MoSSETracker`, `MoSSETrackerDeepFeature`.
`MOSSETracker` works with single and multi-channel;
`MoSSETrackerDeepFeature` work with features extracted by CNN (resnet50).

Choose tracker class at #line 34: tracker = MoSSETracker()

2. Choose the sequence to track
```
#line 20: parser.add_argument('--sequences',nargs="+",default=[4],type=int)   
```