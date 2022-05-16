/*
    System information
*/
var ManifestApi = "/api/manifest";
var ModelUpdate = "/api/model-update";
var StaticInfoApi = "/api/static-info";
var DynamicInfoApi = "/api/dynamic-info";
var ExpertInfoApi = "/api/expert-info";
var RollBackApi = "/api/roll-back";
var ImageInfoApi = "/api/image";
var TSNEUpdateApi = "/api/tsne-update";
var IncreTSNEUpdateApi = "/api/incre-tsne-update";
var PropagationApi = "/api/propagation";
var InstanceCandidateSelectionApi = "/api/instance-candidate-selection";
var WorkerCandidateSelectionApi = "/api/worker-candidate-selection";
var CroppingApi = "/api/cropping-info";
var SetNameSelector = "dataset-select";
var DatasetName = null;

/*
    flag
*/
var FocusedInstanceID = 0;

/*
    View Object
*/
var RankingsView = null;
var InstancesView = null;
var WorkerView = null;
var WorkerDialogView = null;
var ImageView = null;
var DetailView = null;
var TrailView = null;
var MatrixView = null;

/*
    Const variables for data storage
*/
var WorkerLabels = null;
var InstanceTotalNum = null;
var WorkerTotalNum = null;
var LabelTotalNum = null;
var ModelTotalNum = null;
var LabelNames = null;
var KNeighbors = null;
var SimilarityMatrix = null;
var TSNECoordinate = null;
var ImageUrl = null;
var WorkerPredictionMatrix = null;
/*
    Variables for intermediate results
*/

var PosteriorDistribution = null;
var PosteriorLabels = null;
var Uncertainty = null;
var SpammerScore = null;
var SpammerRanking = null;
var SpammerList = null;
var InstanceRanking = null;
var InstanceList = null;
var UncertaintyMax = null;
var WorkerAccuracy = null;
var SelectedList = null;
var SelectedListThisRound = [];
var SelectedGlobal = null;
var GuidedTSNE = null;
var MixedIndex = null;
var Constraint = null;
var Influence = null;
var PrePosteriorLabels = null;
var LabeledList = null;
var TrailLog = [];
var SeedLog = [];
var TrailSpammer = [];
var Seed = 0;
var ThisRoundLabeled = [];
var ChangedWorker = {};

var ValidatedLabel = {};
var ValidatedSpammer = {};

/*
    Menu variables
* */
var InstanceMenu = null;
var SpammerMenu = null;

/*
    Variables for info needed exported
*/
var ExpertLabel = null;
var isSpammer = null;

/*
    color list
*/
var CategoryColor = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#bcbd22",
    "#e377c2",
    "#990099",
    "#17becf",
    "#8c564b"
];

var Gray = "#a8a8a8";

/*
    some variables that debug needed
*/
var AnimationDuration = 2000;

/*
    some variables that debug needed
*/
var ManifestData = null;
var StaticInfoData = null;
var DynamicInfoData = null;
var FeedBack = null;
var TrueLabelOrPosteriorFlag = 0;
var DebugView = null;

/*
    variables that needs careful usage
*/
var TrueLabels = null;

/*
    variables added by fangxin
 */
var SelectedInstances = [];
var SelectedWorkers = [];