# sentimentClassifier

A vanilla Naive Bayes classifier trained with tweets that tells you whether or not you are saying something nice.

Like this:

```bash
Tell me something...
love
That is nice.
```
```bash
Tell me something...
hate
That is not nice.
```
```bash
Tell me something...
homework
That is nice.
```
```bash
Tell me something...
homework on friday
That is not nice.
```

[See the code here.](https://github.com/da-cali/sentimentClassifier/blob/master/src/Main.hs)

Run it by either copying the source code and building it with matrix and random as dependencies; or by cloning this repository: 

0. Install [stack](https://docs.haskellstack.org/en/stable/README/) if necessary.
    
1. Clone repository:
    ```
    git clone https://github.com/da-cali/sentimentClassifier
    ```
2. Open folder:
    ```
    cd sentimentClassifier
    ```
3. Build project:
    ```
    stack build
    ```
4. Train and run:
    ```
    stack runhaskell src/Main.hs
    ```

##### Note: Increase the sample size to train with more examples and get more accurate responses.

### Authors:
#### Louise Brett, Dan Castillo, Michael Ton.