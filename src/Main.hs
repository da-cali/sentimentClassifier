module Main where

import Data.Char
import Data.List
import Data.Maybe
import Data.Matrix
import System.Random
import Control.Monad

main :: IO ()
main = do
  -- Training a Naive Bayes classifier.
  putStrLn "Training model..."
  -- Reading files with tweets. Data from Sentiment140.com.
  niceTweets <- readFile "src/niceTweets.txt"
  meanTweets <- readFile "src/meanTweets.txt"
  -- Getting global random number generator to randomize data set.
  generator <- getStdGen
  let -- Matrix with processed lines (a column vector, but the type is Matrix).
      allExamples :: Matrix (Int,[String])
      allExamples = (niceTweets`labeledWith`1) <-> (meanTweets`labeledWith`0)
      -- Number of examples to train and test the model on.
      sampleSize :: Int
      sampleSize = 1000
      -- List of sampleSize random indices.
      randomIndices :: [Int]
      randomIndices = take sampleSize (randomRs (1,nrows allExamples) generator)
      -- Test and train data sets with a 25-75 split.
      test, train :: [(Int, [String])]
      test = [allExamples!(i,1) | i <- take (sampleSize`div`4) randomIndices]
      train = [allExamples!(i,1) | i <- drop (sampleSize`div`4) randomIndices]
      -- Set of all words from our training examples. 
      vocabulary :: [String]
      vocabulary = nub (concatMap snd train)
      -- List with the number of training examples per label (matching labels).
      examplesPerLabel :: [Double]
      examplesPerLabel = map count [0,1] where
        count l = foldr (\ (n,_) a -> a + if l == n then 1 else 0) 1 train
      -- Matrix of learnt "parameters" (counts).
      trainedParameters :: Matrix Double
      trainedParameters = fit train vocabulary examplesPerLabel
      -- Predictions of our trained model on the test set.
      predictions :: [Int]
      predictions = map (\ e -> predict e trainedParameters vocabulary examplesPerLabel)
                        [unwords example | example <- map snd test]
      -- Percentage of incorrect predictions, where 0 < error < 1.
      error :: Double
      error = fromIntegral (length [(p,l) | (p,l) <- zip predictions (map fst test), p /= l])
            / fromIntegral (length test)
  -- Showing error.
  if error > 1.0 
     then putStrLn "Something went wrong."
     else do putStrLn "Training complete. Error: "
             print error
  -- Running main program.
  forever $ do
    putStrLn "Tell me something..."
    input <- getLine
    putStrLn $ if predict input trainedParameters vocabulary examplesPerLabel == 1
                  then "That is nice." 
                  else "That is not nice."

-- Separates semantically important characters (so they are treated as words) 
-- and removes other punctuation from text to return a set of its words.
clean :: String -> [String]
clean text = (nub.words.filterpunc) text where
  filterpunc [] = []
  filterpunc (h:t) = let processed c | c `elem` "?!@#$%&^*=+-" = ' ' : c : " "
                                     | c `elem` ".,:;_'`()[]{}0123456789" = []
                                     | otherwise = toLower c : []
                     in processed h ++ filterpunc t

-- Processes the file to create a vector of tuples of form (label,sentence).
labeledWith :: String -> Int -> Matrix (Int,[String])
labeledWith text label = fromList (length labeledText) 1 labeledText where
  labeledText = zip [label,label..] [clean line | line <- lines text]

-- "Trains" the model by returning a matrix where the (i,j) element is the 
-- number of jth-labelled examples with ith word. (e.g. number of "Negative"
-- examples with word "sad")
fit :: [(Int,[String])] -> [String] -> [Double] -> Matrix Double
fit examples vocabulary labels = foldr addCount allOnesMatrix examples where
  -- Laplace smoothing is done here by initializing a matrix with 1's.
  allOnesMatrix :: Matrix Double
  allOnesMatrix = matrix (length labels) (length vocabulary) (\ _ -> 1)
  -- Traverses the sentence to update the matrix at every word.
  addCount :: (Int,[String]) -> Matrix Double -> Matrix Double
  addCount (label,sentence) parameters = let  
    add1 word params = setElem (getElem i j params + 1) (i,j) params where
      i = 1 + label
      j = 1 + fromJust (elemIndex word vocabulary)
    in foldr add1 parameters sentence

-- Predicts a label (e.g."Positive") given an input phrase and a trained model.
predict :: String -> Matrix Double -> [String] -> [Double] -> Int
predict phrase parameters vocabulary examplesPerLabel = let
  -- Returns label corresponding to index with the highest probability in list.
  highestProbability :: [Double] -> Int
  highestProbability list = fromJust (elemIndex (maximum list) list)
  -- Computes the probability that the input phrase has the given label.
  -- (Not really conditional probabilities since we do not compute the 
  -- denominator). Using log probabilities to avoid underflow issues.
  probability :: Int -> Double
  probability label = sum (fmap log conditionals) + log labelProbability where
    -- Probability p(y=l) that any phrase is tagged with label l.
    labelProbability :: Double
    labelProbability = examplesPerLabel!!label / sum examplesPerLabel
    -- Vector of conditional probabilities p(x|y) that a y-labeled phrase has
    -- the word x. We speed up computations by mapping ¬p(x|y) to the y row 
    -- and only modify it at the indices of the words from the input phrase.
    conditionals :: Matrix Double
    conditionals = negProbs `positiveAt` wordIndices where
      -- Uses the count to compute the probability ¬p(x|y).
      negProbs = fmap (\ count -> 1 - (count/examplesPerLabel!!label))
                      (submatrix (label+1) (label+1) 1 (length vocabulary) parameters)
      -- Reverses the probability to p(x|y) at every given index.
      positiveAt ps [] = ps
      positiveAt ps (j:js) = positiveAt (setElem (1 - getElem 1 (j+1) ps) (1,j+1) ps) js
      -- Indices of the words from the input phrase that are in our vocabulary.
      wordIndices = map (\ word -> (fromJust . elemIndex word) vocabulary)
                        [word | word <- clean phrase, word `elem` vocabulary]
  in highestProbability (map probability [0 .. nrows parameters - 1])