module Main where

import Data.Char
import Data.List
import Data.Maybe
import Data.Matrix
import Control.Monad
import System.Random
import qualified Data.Vector as V

main :: IO ()
main = do
  -- Training a Naive Bayes classifier.
  putStrLn "Training model..."
  -- Reading files with tweets. Data from Sentiment140.com.
  niceTweets <- readFile "src/niceTweets.txt"
  meanTweets <- readFile "src/meanTweets.txt"
  -- Getting global random number generator to randomize data set.
  generator <- getStdGen
  let -- List of processed lines.
      allExamples :: [(Int,[String])]
      allExamples = niceTweets`labeledWith`[1,1..] ++ meanTweets`labeledWith`[0,0..]
      -- Number of lines extracted from files.
      dataSize :: Int
      dataSize = length allExamples
      -- Matrix with our data (a row vector actually, but the type is Matrix).
      allData :: Matrix (Int, [String])
      allData = fromList 1 dataSize allExamples
      -- Number of examples to train and test the model on.
      sampleSize :: Int
      sampleSize = 1000
      -- List of s random indices.
      randomIndices :: [Int]
      randomIndices = take sampleSize (randomRs (1,dataSize) generator)
      -- Create test and train data sets with a 25-75 split.
      test, train :: [(Int, [String])]
      test = map (\ i -> getElem 1 i allData) (take (sampleSize`div`4) randomIndices)
      train = map (\ i -> getElem 1 i allData) (drop (sampleSize`div`4) randomIndices)
      -- Set of all words from our training examples. 
      vocabulary :: [String]
      vocabulary = makeVocabulary train
      -- Where 0 = Negative and 1 = Positive.
      labels :: [Int]
      labels = [0,1]
      -- List with the number of examples per label (matching labels).
      tweetsPerLabel :: [Double]
      tweetsPerLabel = map count labels where
        count l = foldr (\ (n,_) a -> a + if l == n then 1 else 0 ) 1 train
      -- Matrix of learnt "parameters" (counts). Laplace smoothing is done
      -- here by initializing the matrix with 1's before fitting.
      trainedParameters :: Matrix Double
      trainedParameters = fit (matrix (length labels) (length vocabulary) $ \_-> 1)
                              train vocabulary labels
      -- Predictions of our trained model on the test set.
      predictions :: [Int]
      predictions = map 
        (\e-> predict e trainedParameters vocabulary labels tweetsPerLabel sampleSize)
        (map (\ (_,l) -> ' ' : unwords l) test)
      -- Percentage of incorrect predictions, 0 < error < 1.
      error :: Double
      error = fromIntegral (length $ filter (\(p,(l,_))-> p /= l) (zip predictions test))
            / fromIntegral (length test)
  -- Showing error.
  if error > 1.0 
    then putStrLn "Something went wrong."
    else do putStrLn "Training complete. Error: "
            print error
  -- Running main program.
  forever $ do
    putStrLn "Tell me something..."
    phrase <- getLine
    putStrLn $ (\ p -> if p==1 then "That is nice." else "That is not nice.")
               (predict phrase trainedParameters vocabulary labels tweetsPerLabel sampleSize)

-- Separates semantically important characters and removes all other punctuation from text.
clean :: String -> String
clean text = filter (`notElem` ".,:;-+_='[]{}0123456789") (separate text) where
  separate [] = []
  separate (h:t) | h `elem` "?!@#$%&" = h : ' ' : separate t
                 | otherwise = h : separate t

-- Process the file to create a list tuples of form (label,sentence).
labeledWith :: String -> [Int] -> [(Int,[String])]
labeledWith text labels = zip labels (map (nub.words.clean) (lines $ map toLower text))

-- Creates vocabulary (set of all words in lines).
makeVocabulary :: [(Int,[String])] -> [String]
makeVocabulary lns = nub $ concatMap (\ (m,l) -> l ) lns

-- "Trains" the model by returning a matrix where the (i,j) element is the 
-- number of jth-labelled examples with ith word. (e.g. number of "Negative"
-- examples with word "sad")
fit :: Matrix Double -> [(Int,[String])] -> [String] -> [Int] -> Matrix Double
fit parametersToTrain examples vocabulary labels = let
  addCount (l,sentence) prams = foldr add1 prams sentence where 
    add1 w ps = let i = 1 + fromJust (elemIndex l labels)
                    j = 1 + fromJust (elemIndex w vocabulary)
                in setElem (getElem i j ps + 1) (i,j) ps
  in foldr addCount parametersToTrain examples 

-- Given an input phrase and a trained model it predicts a label (e.g."Positive").
predict :: [Char] -> Matrix Double -> [String] -> [Int] -> [Double] -> Int -> Int
predict phrase parameters vocabulary labels examplesPerLabel numOfExamples = let
  -- Returns the label corresponding to the index of the highest probability in list.
  highestProbability :: [Double] -> Int
  highestProbability list = labels !! fromJust (elemIndex (maximum list) list)
  -- Computes the probability that the input phrase has the given label.
  -- (Not really conditional probabilities since we do not compute the 
  -- denominator). Using log probabilities to avoid underflow issues.
  probability :: Int -> Double
  probability label = sum (map log conditionals) + log labelProbability where
    -- Probability p(y = l) that any phrase is tagged with label l.
    labelProbability :: Double
    labelProbability = examplesPerLabel!!fromJust (elemIndex label labels) 
                     / fromIntegral numOfExamples
    -- Conditional probabilities p(x|y) that a y-labeled phrase has the word x.
    -- We speed up computations by mapping ¬p(x|y) to the matrix and then 
    -- only modify it at the indices of the words from the input phrase.
    conditionals :: [Double]
    conditionals = let
      -- Index of label.
      i = fromJust (elemIndex label labels)
      -- Indices of the words from the input phrase that are in our vocabulary.
      phraseIdxs = map (\ word -> fromJust $ elemIndex word vocabulary)
                       (filter (`elem` vocabulary) $ (nub.words.clean) phrase)
      -- Uses the count to compute the probability ¬p(x|y).
      negProb _ count = 1 - (count/examplesPerLabel!!i)
      -- Reverses the probability to p(x|y) at every given index.
      posProb [] ps = ps
      posProb (j:js) ps = posProb js (setElem (1 - getElem (i+1) (j+1) ps) (i+1,j+1) ps)
      in V.toList $ getRow (i+1) (posProb phraseIdxs $ mapRow negProb (i+1) parameters)
  in highestProbability (map probability labels)