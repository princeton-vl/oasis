# The evaluation script for instance-level semantic labeling.
# We use this script to evaluate your approach on the test set.
#
# Please check the description of the "getPredictionInfoFile" method below
# and set the required environment variables as needed, such that
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# To run this script, make sure that your results contain text files
# (one for each test set image) with the content:
#   relPathPrediction1 labelIDPrediction1 confidencePrediction1
#   relPathPrediction2 labelIDPrediction2 confidencePrediction2
#   ...
# Example:
#   lindau_000014_000019_26000.png 26 1.0
# denotes this a car (26) instance with confidence 1.0
#
# - The given paths "relPathPrediction" point to images that contain
# binary masks for the described predictions, where any non-zero is
# part of the predicted instance.
# - The label IDs "labelIDPrediction" specify the class of that mask,
# encoded as defined in labels.py.
# - The field "confidencePrediction" is a float value that assigns a
# confidence score to the mask.

from __future__ import print_function, absolute_import, division
import os, sys
import fnmatch
from copy import deepcopy
import numpy as np
import argparse

# Cityscapes imports
from csHelpers import *

############################################
# Set up global parameters
############################################
parser = argparse.ArgumentParser(description="PyTorch Object Detection")
parser.add_argument("--gt_list_txt",
                    default='./debug/test7/test7.txt',
                    help="A txt file that lists paths to gt files")
parser.add_argument("--pred_path",
                    default='./debug/test7/pred',
                    help="Path to the folder that contains pred files")
parser.add_argument("--gt_suffix",
                    default='_gtFine_instanceIds.png',
                    help="The suffix that the ground truth should have. For example the suffix of 1.png is .png")
parser.add_argument("--quiet",
                    action="store_true")       #default is False
parser.add_argument("--csv",
                    action="store_true")       #default is False
parser.add_argument("--colorized",
                    action="store_false")        #default is True
parser.add_argument("--output_file", "-o", default=None,
                    help="The record file to output to.")
args = parser.parse_args()

# overlaps for evaluation
args.overlaps           = np.arange(0.5,1.,0.05)
# minimum region size for evaluation [pixels]
args.minRegionSizes     = [100]
# only one class is needed.  Use surface as a dummy label. Modification: Weifeng  TODO
args.instLabels         = ['surface']



import pickle
def save_obj(obj, name, verbal=False ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if verbal:
            print( " Done saving %s" % name)

# Routine to read ground truth image
def readGTImage(gtImageFileName):
    return np.array(Image.open(gtImageFileName))

def getPredictionInfoFile( groundTruthFile, args ):
    # A file matches, if the filename follows the pattern
    #   Prefix*.txt
    # for a ground truth filename
    #   Prefix + args.gt_suffix
    def parse_gtFile(groundTruthFile):
        filename = os.path.basename(groundTruthFile)
        idx = filename.find(args.gt_suffix)
        assert(idx >= 0)
        return filename[:idx]

    prefix = parse_gtFile(groundTruthFile)
    #filePattern = "{}*.txt".format(prefix)
    filePattern = "{}.txt".format(prefix)

    predInfoFile = glob.glob(os.path.join(args.pred_path, filePattern))
    if len(predInfoFile) > 1 or len(predInfoFile) == 0:
        import pdb; pdb.set_trace()
        printError("Error finding predInfoFile for %s" % groundTruthFile)
    predictionFile = predInfoFile[0]

    return predictionFile

def readPredInst(predInfoFileName, ROINp):
    ##############################################
    # Read prediction from info File and create an Instance for each inst
    #   imgFile, predId, confidence
    # return
    #   predInsts: a dictionary whose keys are filepaths to each binary mask for an instance
    predInsts = {}
    if (not os.path.isfile(predInfoFileName)):
        printError("Infofile '{}' for the predictions not found.".format(predInfoFileName))

    with open(predInfoFileName, 'r') as f:
        for line in f:
            splittedLine = line.split(" ")
            assert(len(splittedLine) == 3)

            predImgFile = os.path.join( os.path.dirname(predInfoFileName),splittedLine[0])
            labelID = int(float(splittedLine[1].strip()))
            conf = float(splittedLine[2].strip())

            ##############################################
            # create instance
            predImg = Image.open(predImgFile)
            predImg = predImg.convert("L")      # https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes
            predNp    = np.array(predImg)
            # make the image really binary, i.e. everything non-zero is part of the prediction
            boolPredInst   = predNp != 0
            boolROI = ROINp != 0
            # Consider the ROI as well
            predPixelCount = np.count_nonzero( np.logical_and(boolPredInst, boolROI) )

            # skip if actually empty
            if not predPixelCount:
                continue
            predInstance = {}
            predInstance["imgName"]          = predImgFile
            predInstance["labelID"]          = labelID
            predInstance["pixelCount"]       = predPixelCount
            predInstance["confidence"]       = conf

            predInsts[predImgFile]   = predInstance

    return predInsts


def getGtInstances(groundTruthList, ROIImgList, args):
    ##################################################
    # Compute a dictionary of all ground truth instances
    # return
    #   gtInstances: a dictionary, whose keys are the filepath to gt instance seg images
    print("Creating ground truth instances from png files.")
    gtInstances = {}

    for imageFileName, ROIFileName in zip(groundTruthList, ROIImgList):
        # Load image
        imgNp = np.array(Image.open(imageFileName))
        ROINp = np.array(Image.open(ROIFileName))
        ROINp = ROINp[:, :, 0]
        #import pdb; pdb.set_trace()
        boolROI = ROINp != 0
        # Initialize label categories
        instances = {'surface':[]}  # TODO

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId == 0:
                continue
            # Corsider the ROI as well
            pixelCount = np.count_nonzero( np.logical_and(imgNp == instanceId, boolROI) )

            _inst = {}
            _inst["instID"]     = instanceId
            _inst["labelID"]    = 1             # TODO
            _inst["pixelCount"] = pixelCount

            instances['surface'].append(_inst)
            assert(pixelCount > 0)

        gtInstances[imageFileName] = instances

    return gtInstances

def matchGtWithPreds(predictionInfoList,
                     groundTruthList,
                     ROIImgList,
                     gtInstances,
                     args):
    ##################################################################
    # match ground truth instances with predicted instances
    # return
    #   matches: A dict whose keys are are the filepath to each gt instance seg image.
    #            Each elem has two keys "groundTruth" and "prediction".
    #            "prediction": points to a gt instance
    #            "prediction": points to the list of prediction that are matched to this gt inst.
    matches = {}
    print("Matching {} pairs of images...".format(len(predictionInfoList)))

    count = 0
    for (predInfo,gt,ROI) in zip(predictionInfoList,groundTruthList,ROIImgList):
        # Read input files
        gtImgNp  = readGTImage(gt)
        ROIImgNp = readGTImage(ROI)
        ROIImgNp = ROIImgNp[:, :, 0]
        predInst = readPredInst(predInfo, ROIImgNp)

        # Get and filter ground truth instances
        curGtInstancesOrig = gtInstances[ gt ]
        # Try to assign all predictions    Important: Weifeng
        (curGtInstances,curPredInstances) = assignGt2Preds(curGtInstancesOrig, gtImgNp, ROIImgNp, predInst)

        # append to global dict
        matches[ gt ] = {}
        matches[ gt ]["groundTruth"] = curGtInstances
        matches[ gt ]["prediction"]  = curPredInstances

        count += 1
        print("\rImages Processed: {}".format(count), end=' ')
        sys.stdout.flush()

    return matches


def assignGt2Preds(gtInstancesOrig, gtNp, ROINp, predInst):
    ################################################################################
    # For a given frame, assign all predicted instances to ground truth instances
    # Return two lists
    #  - predInstances: contains all predictions and their associated gt
    #  - gtInstances:   contains all gt instances and their associated predictions

    predInstances = {}
    predInstances['surface'] = []       # TODO

    # We already know about the gt instances
    # Add the matching information array
    gtInstances = deepcopy(gtInstancesOrig)
    for gt in gtInstances['surface']:   # TODO
        gt["matchedPred"] = []

    # Get a mask of void labels in the groundtruth  TODO
    voidLabelIDList = []
    # for label in labels:
    #     if label.ignoreInEval:
    #         voidLabelIDList.append(label.id)
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.in1d.html
    boolVoid = np.in1d(gtNp, voidLabelIDList).reshape(gtNp.shape)           # TODO
    boolROIImage = ROINp > 0


    # Loop through all prediction masks
    for predImageFile in predInst:
        # The predicted Instance
        predInstance = deepcopy(predInst[predImageFile])

        # A list of all overlapping ground truth instances for this image
        matchedGt = []

        # Read the mask
        predImage = Image.open(predImageFile)
        predImage = predImage.convert("L")      # https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes
        predNp    = np.array(predImage)
        boolPredInst = predNp != 0
        # Determine the number of pixels overlapping void       TODO   I am not really sure what this void is for????
        predInstance["voidIntersection"] = np.count_nonzero( np.logical_and(boolVoid, boolPredInst) )

        # Loop through all ground truth instances with matching label
        for (gtNum,gtInstance) in enumerate(gtInstancesOrig['surface']):        # TODO
            # Modification: Weifeng      consider the ROI when getting the intersection
            intersection = np.count_nonzero( np.logical_and( boolROIImage,
                                                             np.logical_and(gtNp == gtInstance["instID"], boolPredInst)) )

            # If they intersect add them as matches to both dicts
            if (intersection > 0):
                gtCopy   = gtInstance.copy()
                predCopy = predInstance.copy()

                # let the two know their intersection
                gtCopy["intersection"]   = intersection
                predCopy["intersection"] = intersection

                # append ground truth to matches
                matchedGt.append(gtCopy)
                # append prediction to ground truth instance    TODO
                gtInstances['surface'][gtNum]["matchedPred"].append(predCopy)

        predInstance["matchedGt"] = matchedGt
        predInstances['surface'].append(predInstance)   # TODO

    return (gtInstances,predInstances)

def evaluateMatches(matches, args):
    ##########################################################################
    # Important: please read!
    ##########################################################################
    # In the end, we need two vectors for each class and for each overlap
    # The first vector (y_true) is binary and is 1, where the ground truth says true,
    # and is 0 otherwise.
    # The second vector (y_score) is float [0...1] and represents the confidence of
    # the prediction.
    #
    # We represent the following cases as:
    #                                       | y_true |   y_score
    #   gt instance with matched prediction |    1   | confidence
    #   gt instance w/o  matched prediction |    1   |     0.0          I think the code just discard this case.
    #          false positive prediction    |    0   | confidence
    #
    # The current implementation makes only sense for an overlap threshold >= 0.5,
    # since only then, a single prediction can either be ignored or matched, but
    # never both. Further, it can never match to two gt instances.
    # For matching, we vary the overlap and do the following steps:
    #   1.) remove all predictions that satisfy the overlap criterion with an ignore region (either void or *group)     # Important
    #   2.) remove matches that do not satisfy the overlap              # Important: I think this refers to throwing away "gt instance w/o  matched prediction"??
    #   3.) mark non-matched predictions as false positive
    overlaps  = args.overlaps
    ap = np.zeros((1, len(args.instLabels), len(overlaps)), np.float)
    # loop through all configs
    for (dI, minRegionSize) in enumerate(args.minRegionSizes):
        for (oI,overlapTh) in enumerate(overlaps):
            for (lI,labelName) in enumerate(['surface']):       # iterate through all semantic classes
                # vars for summarization
                y_true   = np.empty( 0 )        # This var accumulates the true and false positives (curTrues) over all the test images, 1 denotes true positive, 0 denotes false positive
                y_score  = np.empty( 0 )        # This var accumulates the confidence scores (curScore) over all the test images
                haveGt   = False                # found at least one gt instance over all the test images?
                havePred = False                # found at least one predicted instance over all the test images?

                # Count hard false negatives.
                hardFns  = 0

                # iterate through all gt images
                for img in matches:
                    predInstances = matches[img]["prediction" ][labelName]
                    gtInstances   = matches[img]["groundTruth"][labelName]
                    # filter groups in ground truth
                    gtInstances   = [gt for gt in gtInstances if gt["pixelCount"]>=minRegionSize]   # Modification: Weifeng

                    if gtInstances:
                        haveGt = True
                    if predInstances:
                        havePred = True

                    curTrue  = np.ones(len(gtInstances))     # 1 denotes that the match to this gt instance is a true positive, 0 denotes false positive
                    curScore = np.ones(len(gtInstances)) * (-float("inf"))
                    curMatch = np.zeros(len(gtInstances), dtype=np.bool) # 0 means that this gt instance does not have a match

                    # collect matches, iterate through all gt instance of this class
                    for (gtI,gt) in enumerate(gtInstances):
                        foundMatch = False
                        for pred in gt["matchedPred"]:
                            overlap = float(pred["intersection"]) / (gt["pixelCount"]+pred["pixelCount"]-pred["intersection"])

                            # Important: Weifeng,   only process instances with larger overlap
                            if overlap > overlapTh:
                                # the score
                                confidence = pred["confidence"]

                                # Important: Weifeng
                                # if we already had a prediction for this groundtruth
                                # the prediction with the lower score is automatically a false positive
                                if curMatch[gtI]:
                                    maxScore = max( curScore[gtI] , confidence )
                                    minScore = min( curScore[gtI] , confidence )
                                    curScore[gtI] = maxScore
                                    # append false positive
                                    curTrue  = np.append(curTrue,0)         # append 0 to denotes false positive
                                    curScore = np.append(curScore,minScore)
                                    curMatch = np.append(curMatch,True)     # append True to denotes a match
                                # otherwise set score
                                else:
                                    foundMatch = True
                                    curMatch[gtI] = True
                                    curScore[gtI] = confidence

                        # Weifeng:
                        # I think hard false negative refers to the gt instances that are not assigned
                        # any predictions during the matching phase. It counts as a false negative.
                        if not foundMatch:
                            hardFns += 1

                    # Important: Weifeng
                    # remove non-matched ground truth instances             ###### Weifeng: throw away unmatched gt?
                    curTrue  = curTrue [curMatch==True]
                    curScore = curScore[curMatch==True]

                    # Important: Weifeng
                    # collect non-matched predictions as false positive
                    for pred in predInstances:
                        foundGt = False
                        for gt in pred["matchedGt"]:
                            overlap = float(gt["intersection"]) / (gt["pixelCount"]+pred["pixelCount"]-gt["intersection"])
                            if overlap > overlapTh:     #matched
                                foundGt = True
                                break
                        if not foundGt:     #non matched
                            # collect number of void and *group pixels
                            nbIgnorePixels = pred["voidIntersection"]
                            for gt in pred["matchedGt"]:
                                # small ground truth instances
                                if gt["pixelCount"] < minRegionSize:    # Modification: Weifeng
                                    nbIgnorePixels += gt["intersection"]
                            proportionIgnore = float(nbIgnorePixels)/pred["pixelCount"]

                            # # Important: Weifeng
                            # if not ignored
                            # append false positive
                            if proportionIgnore <= overlapTh:
                                curTrue = np.append(curTrue,0)
                                confidence = pred["confidence"]
                                curScore = np.append(curScore,confidence)

                    # append to overall results
                    y_true  = np.append(y_true,curTrue)
                    y_score = np.append(y_score,curScore)

                #Important: Weifeng
                # compute the average precision, defined as the area under the PR-curve
                if haveGt and havePred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    scoreArgSort      = np.argsort(y_score)
                    yScoreSorted      = y_score[scoreArgSort]
                    yTrueSorted       = y_true[scoreArgSort]
                    yTrueSortedCumsum = np.cumsum(yTrueSorted)

                    # unique threshhold (confidence score)
                    (thresholds,uniqueIndices) = np.unique(yScoreSorted , return_index=True )

                    # since we need to add an artificial point to the precision-recall curve
                    # increase its length by 1
                    nbPrecRecall = len(uniqueIndices) + 1

                    # prepare precision recall
                    nbExamples     = len(yScoreSorted)
                    nbTrueExamples = yTrueSortedCumsum[-1]

                    precision      = np.zeros(nbPrecRecall)
                    recall         = np.zeros(nbPrecRecall)

                    # deal with the first point
                    # only thing we need to do, is to append a zero to the cumsum at the end.
                    # an index of -1 uses that zero then
                    yTrueSortedCumsum = np.append( yTrueSortedCumsum , 0 )

                    # deal with remaining
                    for idxRes,idxScores in enumerate(uniqueIndices):
                        cumSum = yTrueSortedCumsum[idxScores-1]
                        tp = nbTrueExamples - cumSum
                        fp = nbExamples     - idxScores - tp
                        fn = cumSum + hardFns                   # false negative
                        p  = float(tp)/(tp+fp)
                        r  = float(tp)/(tp+fn)
                        precision[idxRes] = p
                        recall   [idxRes] = r # recall is sorted

                    # first point in curve is artificial        # Weifeng: pay attention here.
                    precision[-1] = 1.
                    recall   [-1] = 0.

                    # compute average of precision-recall curve
                    # integration is performed via zero order, or equivalently step-wise integration
                    # first compute the widths of each step:
                    # use a convolution with appropriate kernel, manually deal with the boundaries first
                    recallForConv = np.copy(recall)
                    recallForConv = np.append( recallForConv[0] , recallForConv )
                    recallForConv = np.append(recallForConv, 0.            )

                    stepWidths = np.convolve(recallForConv,[-0.5,0,0.5],'valid')

                    # integrate is now simply a dot product
                    apCurrent = np.dot(precision , stepWidths )
                elif haveGt:
                    apCurrent = 0.0
                else:
                    apCurrent = float('nan')
                ap[dI,lI,oI] = apCurrent
    return ap

def computeAverages(aps,args):
    ########################################
    # Compute an average ap over all overlap thresholds, one ap for each semantic classes
    # Input
    #   aps: np matrix of (1, N_labels, N_overlap_thresh)    N_labels = 1, N_overlap_thresh = 10

    o50  = np.where(np.isclose(args.overlaps,0.5  ))
    o75  = np.where(np.isclose(args.overlaps,0.75  ))
    o90  = np.where(np.isclose(args.overlaps,0.90  ))

    avgDict = {}
    avgDict["allAp"]       = np.nanmean(aps[ 0,:,:  ])
    avgDict["allAp50%"]    = np.nanmean(aps[ 0,:,o50])
    avgDict["allAp75%"]    = np.nanmean(aps[ 0,:,o75])
    avgDict["allAp90%"]    = np.nanmean(aps[ 0,:,o90])

    avgDict["classes"]  = {}
    for (lI,labelName) in enumerate(args.instLabels):
        avgDict["classes"][labelName]             = {}
        avgDict["classes"][labelName]["ap"]       = np.average(aps[ 0,lI,  :])
        avgDict["classes"][labelName]["ap50%"]    = np.average(aps[ 0,lI,o50])
        avgDict["classes"][labelName]["ap75%"]    = np.average(aps[ 0,lI,o75])
        avgDict["classes"][labelName]["ap90%"]    = np.average(aps[ 0,lI,o90])

    return avgDict

def printResults(avgDict, args):
    sep     = (","         if args.csv       else "")
    col1    = (":"         if not args.csv   else "")
    noCol   = (colors.ENDC if args.colorized else "")
    bold    = (colors.BOLD if args.colorized else "")
    lineLen = 90

    print("")
    if not args.csv:
        print("#"*lineLen)
    line  = bold
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_75%"    ) + sep
    line += "{:>15}".format("AP_90%"    ) + sep

    line += noCol
    print(line)
    if not args.csv:
        print("#"*lineLen)

    allApAvg  = avgDict["allAp"]
    allAp50o  = avgDict["allAp50%"]
    allAp75o  = avgDict["allAp75%"]
    allAp90o  = avgDict["allAp90%"]

    line  = "{:<15}".format("average") + sep + col1
    line += getColorEntry(allApAvg , args) + sep + "{:>15.3f}".format(allApAvg)  + sep
    line += getColorEntry(allAp50o , args) + sep + "{:>15.3f}".format(allAp50o)  + sep
    line += getColorEntry(allAp75o , args) + sep + "{:>15.3f}".format(allAp75o)  + sep
    line += getColorEntry(allAp90o , args) + sep + "{:>15.3f}".format(allAp90o)  + sep

    line += noCol
    print(line)
    print("")

def prepareJSONDataForResults(avgDict, aps, args):
    JSONData = {}
    JSONData["averages"] = avgDict
    JSONData["overlaps"] = args.overlaps.tolist()
    JSONData["minRegionSizes"]      = args.minRegionSizes
    JSONData["instLabels"] = args.instLabels
    JSONData["resultApMatrix"] = aps.tolist()

    return JSONData

########################################################################
# The backbone of the evaluation pipeline
########################################################################
def evaluateImgLists(predictionInfoList, groundTruthList, ROIImgList, args):
    try:
        # get dictionary of all ground truth instances
        gtInstances = getGtInstances(groundTruthList, ROIImgList, args)
        # match predictions and ground truth
        matches = matchGtWithPreds(predictionInfoList,groundTruthList,ROIImgList,gtInstances,args)
        # evaluate matches
        apScores = evaluateMatches(matches, args)
        # averages
        avgDict = computeAverages(apScores,args)
        # result dict
        resDict = prepareJSONDataForResults(avgDict, apScores, args)
    except Exception as e:
        printError(str(e))
        return {"run_log": str(e)}

    if not args.quiet:
         # Print results
        printResults(avgDict, args)

    return resDict


# The main method
def main():
    global args

    # read in the list of gt images
    try:
        with open(args.gt_list_txt) as f:
            groundTruthImgList = []
            ROIImgList         = []
            for paths in f:
                splits = paths.strip().split(',')
                groundTruthImgList.append(splits[0])
                ROIImgList.append(splits[1])
    except:
        printError("Error reading ground truth image list from %s" %args.gt_list_txt)
        return {"run_log": "Error reading ground truth image list from %s" %args.gt_list_txt}

    # get the corresponding prediction for each ground truth image
    predictionInfoList = []
    for gt in groundTruthImgList:
        predictionInfoList.append( getPredictionInfoFile(gt,args) )
    # evaluate
    resDict = evaluateImgLists(predictionInfoList, groundTruthImgList, ROIImgList, args)
    resDict["run_log"] = "success"
    
    if rags.output_file:
        save_obj(resDict, args.output_file)

# call the main method
if __name__ == "__main__":
    main()
