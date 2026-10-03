/**
 * Wave 3, group nn-training: automatic gradient tracking for plain tensors, the layer dtype
 * rule, PyTorch weight initialization, Embedding paddingIdx, conv shape errors and
 * tripletMarginLoss. Reference values come from PyTorch 2.12 (float64 unless noted).
 */
import { describe, expect, it } from "vitest";
import { InvalidParameterError, ShapeError } from "../../src/core";
import {
  type AnyTensor,
  dot,
  GradTensor,
  noGrad,
  randn,
  type Tensor,
  tensor,
  transpose,
} from "../../src/ndarray";
import {
  AdaptiveAvgPool2d,
  AlphaDropout,
  AvgPool1d,
  AvgPool2d,
  AvgPool3d,
  BatchNorm1d,
  BatchNorm2d,
  Conv1d,
  Conv2d,
  Conv3d,
  ConvTranspose1d,
  ConvTranspose2d,
  Dropout,
  Dropout2d,
  ELU,
  Embedding,
  EmbeddingBag,
  Flatten,
  FullTransformer,
  GELU,
  GLU,
  GRU,
  GroupNorm,
  Hardswish,
  Identity,
  InstanceNorm1d,
  LayerNorm,
  Linear,
  LocalResponseNorm,
  LogSoftmax,
  LSTM,
  MaxPool1d,
  MaxPool2d,
  MaxPool3d,
  Mish,
  type Module,
  MultiheadAttention,
  mseLoss,
  PositionalEncoding,
  PReLU,
  ReflectionPad2d,
  ReLU,
  RMSNorm,
  RNN,
  SELU,
  Sequential,
  Sigmoid,
  Softmax,
  Softmin,
  Softplus,
  Softsign,
  SpectralNorm,
  Tanh,
  TransformerDecoderLayer,
  TransformerEncoderLayer,
  tripletMarginLoss,
  Unflatten,
  Upsample,
  ZeroPad2d,
} from "../../src/nn";
import { Adam } from "../../src/optim";
import { setSeed } from "../../src/random";

const REF = {
  linear: {
    W: [
      [0.0403250754, -0.3477921188, 0.1838418245],
      [0.1811612844, -0.3085803688, -0.0865316316],
    ],
    b: [-0.3382279277, 0.1498066485],
    x: [
      [1.6907901764, -0.8948279023, -0.3556250334],
      [1.2323857546, 0.1381726563, -1.6821985245],
      [0.3176783025, 0.1328069717, 0.1373240948],
      [0.2405461222, 1.3954508305, 1.3470226526],
    ],
    y: [
      [2.4382081032, 0.2027582824],
      [2.4505412579, 2.0256018639],
      [1.7791550159, -0.9179307222],
      [-0.4578188956, -0.7244732976],
    ],
    pred: [
      [-0.0242113492, 0.763011507],
      [-0.6458456861, 0.4759932486],
      [-0.3463608321, 0.1544931555],
      [-0.566215586, -0.3537845069],
    ],
    loss: 3.02292035232429,
    gW: [
      [-2.1701706127, 0.3355150124, 1.4116345528],
      [-0.1331478465, -0.013935102, 0.7635262727],
    ],
    gb: [-1.9481797337, 0.1134393195],
  },
  embedding: {
    idx: [0, 2, 2, 4, 1],
    out: [
      [1.0, 2.0, 3.0],
      [7.0, 8.0, 9.0],
      [7.0, 8.0, 9.0],
      [13.0, 14.0, 15.0],
      [4.0, 5.0, 6.0],
    ],
    G: [
      [-0.9224173427, -1.8941159248, 1.0056394339],
      [-0.6948469877, 0.9062281847, 0.1072414592],
      [0.612516582, 0.3296454251, -0.8762796521],
      [-1.6768245697, -0.7246872783, 0.9633941054],
      [0.1342400163, 0.5485481024, 2.134853363],
    ],
    grad: [
      [-0.9224173427, -1.8941159248, 1.0056394339],
      [0.1342400163, 0.5485481024, 2.134853363],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [-1.6768245697, -0.7246872783, 0.9633941054],
    ],
  },
  embedding2d: {
    idx: [
      [0, 2],
      [3, 2],
    ],
    out: [
      [
        [1.0, 2.0, 3.0],
        [7.0, 8.0, 9.0],
      ],
      [
        [10.0, 11.0, 12.0],
        [7.0, 8.0, 9.0],
      ],
    ],
  },
  act: {
    x: [
      [-0.87817067, -2.08262038, 1.83170402, -0.55354786],
      [1.03947973, -1.26009333, 0.41556016, -1.30311251],
      [0.43499935, -1.14980733, -0.81498492, -0.91179252],
    ],
    elu: [
      [-0.5844576, -0.8753967, 1.831704, -0.4250935],
      [1.0394797, -0.7163724, 0.4155602, -0.7283152],
      [0.4349993, -0.6833022, -0.557354, -0.5981967],
    ],
    gelu: [
      [-0.1668969, -0.0386566, 1.770304, -0.1605228],
      [0.8841285, -0.1310454, 0.2747323, -0.1256789],
      [0.2906643, -0.1440552, -0.1692311, -0.1651003],
    ],
    softmax: [
      [0.0564629, 0.0169308, 0.8484895, 0.0781168],
      [0.5772939, 0.0579035, 0.3093372, 0.0554654],
      [0.5709164, 0.1170307, 0.1635728, 0.1484801],
    ],
    logsoftmax: [
      [-2.8741722, -4.0786219, -0.1642975, -2.5495496],
      [-0.5494038, -2.8489766, -1.1733234, -2.8919959],
      [-0.5605125, -2.1453192, -1.8104968, -1.9073044],
    ],
    softplus: [
      [0.3475128, 0.1174303, 1.9802451, 0.4541959],
      [1.342276, 0.2496901, 0.9223599, 0.2403427],
      [0.9341158, 0.2751269, 0.3664789, 0.3377595],
    ],
    tanh: [
      [-0.7055017, -0.9694228, 0.9499925, -0.5031745],
      [0.7776825, -0.8510898, 0.3931835, -0.8625223],
      [0.4094909, -0.8176903, -0.672331, -0.7219915],
    ],
    selu: [
      [-1.0275346, -1.5390344, 1.9245733, -0.7473566],
      [1.0921824, -1.2594539, 0.4366295, -1.2804505],
      [0.4570543, -1.2013131, -0.9798836, -1.0516891],
    ],
    hardswish: [
      [-0.3105547, -0.3184256, 1.475042, -0.2257047],
      [0.6998262, -0.3654075, 0.2365618, -0.3685392],
      [0.2490371, -0.3545609, -0.2967924, -0.3173353],
    ],
    softsign: [
      [-0.467567, -0.6756006, 0.6468557, -0.3563121],
      [0.5096789, -0.5575404, 0.2935659, -0.565805],
      [0.3031356, -0.534842, -0.4490312, -0.4769307],
    ],
    sigmoid: [
      [0.293557, 0.1107975, 0.8619645, 0.3650417],
      [0.7387496, 0.2209578, 0.6024203, 0.2136417],
      [0.6070668, 0.2405243, 0.3068292, 0.2866332],
    ],
    mish: [
      [-0.2934564, -0.2434448, 1.7632074, -0.2354471],
      [0.9066527, -0.3082533, 0.3021172, -0.3072992],
      [0.3186401, -0.3085956, -0.2859848, -0.2967663],
    ],
    softmin: [
      [0.1951498, 0.6508096, 0.0129863, 0.1410543],
      [0.0430212, 0.4289185, 0.0802875, 0.4477728],
      [0.075679, 0.3691885, 0.2641415, 0.290991],
    ],
  },
  conv2d: {
    W: [
      [
        [
          [-0.3305178583, 0.1900501698],
          [0.3057141304, 0.1156205609],
        ],
        [
          [-0.2187607288, 0.1864851713],
          [0.2934089005, -0.0723317862],
        ],
      ],
      [
        [
          [-0.057956174, 0.1223713905],
          [0.0454840362, -0.116853483],
        ],
        [
          [-0.3282268941, -0.2479112595],
          [0.0191464704, -0.0189658292],
        ],
      ],
      [
        [
          [0.2919518054, 0.0121000651],
          [-0.091034241, -0.3454549015],
        ],
        [
          [-0.3250670135, 0.1468663812],
          [0.3149107397, -0.0053644087],
        ],
      ],
    ],
    b: [-0.3160065413, 0.1255254149, 0.2879609764],
    x: [
      [
        [
          [0.0591482371, 0.682143569, -1.0268366337, -0.0790459067],
          [0.1438226551, 1.402538538, -0.4166966677, 0.1654539108],
          [0.6057466269, -0.6734573245, -0.7124640942, -0.6910300255],
          [0.2311467379, -0.7555767298, -0.1197852045, 0.9544740915],
        ],
        [
          [-0.6238737702, -0.4702788889, -0.8482166529, 0.8185013533],
          [0.206080094, 0.7011038661, -0.6909569502, 0.2921862006],
          [-0.932921946, 0.3072794676, 0.7593231797, 1.5333269835],
          [-0.7754535675, -2.1510300636, -1.2796473503, -0.9891459346],
        ],
      ],
      [
        [
          [3.7882049084, -0.357550174, 0.1863798499, -1.1653892994],
          [0.8848882318, 1.2843332291, -0.836807847, -1.4269634485],
          [0.9714289904, -2.6343274117, -0.0231094286, 0.7387539744],
          [0.6625412703, 0.752641499, 0.5757877827, 0.0678625405],
        ],
        [
          [-1.1548826694, 0.6102767587, -0.9058299661, -0.8807307482],
          [-0.8606747389, -0.6743758321, -3.020595789, 0.4014256895],
          [0.5376332402, -1.1322522163, -1.8342213631, -0.5264720917],
          [1.1478374004, -0.2811804116, -0.6900880337, 1.4809904099],
        ],
      ],
    ],
    out: [
      [
        [
          [0.058749002, -0.1556335128, 0.0144260198],
          [-0.1999604524, -1.3940166551, -0.1270173879],
          [-0.4714500623, -0.9378512607, -0.3221985851],
        ],
        [
          [0.36023066, 0.4639900013, 0.1937979438],
          [0.1299256868, -0.0214727628, 0.3580778532],
          [0.3627929109, -0.2490153695, -0.6698272036],
        ],
        [
          [0.0107452458, 0.7437494639, 0.1447749226],
          [0.2649697513, 0.7631326654, 0.9702992462],
          [0.8123678592, -0.4659985154, -0.6665341522],
        ],
      ],
      [
        [
          [-1.0543105008, -0.1483329055, -1.9012898693],
          [-0.0698280565, -2.3228698669, 0.0032804886],
          [-0.819803779, 0.7200115094, 0.0093803334],
        ],
        [
          [-0.0235285975, 0.3938881766, 0.5510112003],
          [1.0648710058, 0.8148664958, 0.7788159505],
          [-0.1794126916, 1.0763876213, 0.9267844575],
        ],
        [
          [1.0629988287, -0.1695914566, 0.1091375468],
          [1.7395683984, 0.3294707091, 0.2393448276],
          [0.2412963281, -0.7350125768, 0.5079562959],
        ],
      ],
    ],
    G: [
      [
        [
          [-0.3375511691, -0.6896840578, 1.29601011],
          [0.1465738015, -1.546062001, -1.0087818016],
          [-0.0233760018, -0.8721088386, 0.0627997789],
        ],
        [
          [-1.2316421885, -0.7426664759, 1.3540482844],
          [1.6761834625, -1.0031820183, -1.3088617872],
          [-0.0667343717, 0.0498996948, -0.2552406904],
        ],
        [
          [-0.3744578406, 2.0529082417, -0.6079194933],
          [-0.1631498618, 0.8948142511, 1.3979287703],
          [0.7858575432, 1.1624632624, -1.4806573912],
        ],
      ],
      [
        [
          [-0.2185722055, 0.2038813235, 1.7098205118],
          [-0.0896015096, -0.0688383925, 0.5458745538],
          [-0.1691014596, -0.1442560496, -0.0353219885],
        ],
        [
          [0.3966216871, 0.4870278243, 0.2455951898],
          [0.5294571199, -0.1531746198, 1.1659546714],
          [-2.2938220258, 0.045695657, -0.9577061153],
        ],
        [
          [0.6813590501, -0.0683880148, 0.6412536704],
          [0.4644352171, -0.7987515371, 0.6987273283],
          [-2.1317614547, -0.0583729374, 0.9417890635],
        ],
      ],
    ],
    gW: [
      [
        [
          [-4.0099140908, -0.6380752886],
          [-0.5827783978, -0.5526532696],
        ],
        [
          [-3.6796463509, 1.2458838603],
          [-7.2273971127, -0.6999608883],
        ],
      ],
      [
        [
          [-4.1388566153, 6.6948180197],
          [0.4429802077, -3.4463320531],
        ],
        [
          [-3.3865324123, 6.6051928537],
          [-9.3819759599, -5.2534293544],
        ],
      ],
      [
        [
          [2.9862351468, 2.5569581413],
          [2.465639766, -7.3668485892],
        ],
        [
          [-8.2771869542, 0.7847466463],
          [-3.6295800854, 1.8111494251],
        ],
      ],
    ],
    gb: [-1.2382953956, -2.0625467019, 4.0380778671],
    gx: [
      [
        [
          [0.0736243159, 0.6509448841, -0.8814285156, 0.4046478474],
          [-0.3183480708, 0.8641250278, 0.2226340038, -0.133342434],
          [0.3769278788, -0.1006617801, -1.4496562943, -0.4838273051],
          [-0.0817216676, -0.6365532593, -0.3658633732, 0.5485870625],
        ],
        [
          [0.5998249219, -0.0852994425, -0.1733338689, -0.1832800843],
          [-0.7697413588, 0.4441297387, 0.5553700583, 0.2255052208],
          [-0.2047171866, -0.3084388202, 0.7923049159, -0.0521789469],
          [0.2393385255, 0.109883913, -0.3968369856, 0.0082412826],
        ],
      ],
      [
        [
          [0.2481793465, -0.1003387407, -0.2946263516, 0.3627647147],
          [0.0237146297, -0.3644757485, 0.3803051202, 0.202345222],
          [-0.4791295128, -0.4986098706, 0.7617844504, -0.4270233925],
          [0.0380342963, 0.948208177, -0.1419472679, -0.217518301],
        ],
        [
          [-0.303854241, -0.2212448885, -0.7558656813, 0.3521489642],
          [-0.1471231862, 0.2974668375, -0.1367207962, -0.2164084621],
          [1.6129551598, -0.0211385302, 0.3838185316, 0.3038106162],
          [-0.7648490453, 0.0073379238, 0.277759791, 0.0156664516],
        ],
      ],
    ],
  },
  layernorm: {
    x: [
      [0.5607172847, 0.0789156482, -0.4917934239, 0.4585404694],
      [0.6356552839, -0.151277408, 1.4600994587, 0.4766229987],
      [0.4533816576, -0.6185150146, 1.5127317905, 0.1304706335],
    ],
    out: [
      [0.9916349607, -0.1761609733, -1.5594516957, 0.7439777084],
      [0.0528729116, -1.3166842951, 1.4877141282, -0.2239027447],
      [0.1094756152, -1.2897660225, 1.4923391442, -0.3120487368],
    ],
  },
  triplet: {
    a: [
      [-1.727725625, -1.4914875031, -0.5931913257, -1.1047185659],
      [1.2251092196, 1.1350097656, 1.5602525473, -1.0489749908],
      [-0.4817394316, 0.1301547438, -1.7351869345, -0.7215316892],
      [0.4899295866, 0.3788657188, 0.6752064824, -0.9150240421],
      [-0.183076933, 0.5035846233, 1.4905604124, -0.3858908415],
    ],
    p: [
      [-1.5088957071, -1.8724789143, -0.724632138, -0.3537238598],
      [1.0447928369, 1.4122909725, 1.2276649237, -0.9221411586],
      [-0.3531937957, 0.0068153828, -2.1862377167, -0.5270519555],
      [0.5713305086, 0.090807128, -0.0775408149, -1.2598662853],
      [-0.225846611, 0.7108515382, 1.2696299672, -0.1917468846],
    ],
    n: [
      [-1.9457752705, 0.1226074472, 0.4837569594, -1.340429306],
      [-0.1831595898, -1.6581552029, 1.2270793915, -1.7875320911],
      [0.1695965379, -1.0703936815, -0.7628164291, 2.0018150806],
      [-0.9194440246, 0.742881, 0.2830835879, 0.6941064],
      [-0.1147259027, 0.0902773142, 0.0998177528, 0.4821599126],
    ],
    default: {
      loss: 0.7517746114,
      ga: [
        [-0.0719101133, 0.2507299737, 0.1393886694, -0.1946596404],
        [0.0, -0.0, 0.0, -0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.1463270668, 0.0984539476, 0.1354290054, 0.224290114],
        [0.0316867617, -0.163255694, -0.0424347624, -0.0045596855],
      ],
      gp: [
        [0.0497367414, -0.0865942271, -0.0298748789, 0.1706903278],
        [-0.0, 0.0, -0.0, 0.0],
        [0.0, -0.0, -0.0, 0.0],
        [0.0184910167, -0.0654363612, -0.1709962489, -0.0783354824],
        [-0.023607999, 0.1144039269, -0.1219468475, 0.1071604884],
      ],
      gn: [
        [0.0221733719, -0.1641357466, -0.1095137905, 0.0239693126],
        [0.0, 0.0, 0.0, 0.0],
        [-0.0, 0.0, -0.0, -0.0],
        [0.1278360502, -0.0330175863, 0.0355672434, -0.1459546315],
        [-0.0080787627, 0.048851767, 0.16438161, -0.1026008029],
      ],
    },
    swap: {
      loss: 1.2197568736,
      ga: [
        [-0.0719101133, 0.2507299737, 0.1393886694, -0.1946596404],
        [-0.0129438628, -0.2870106096, 0.1162739364, -0.0979175176],
        [-0.0492036878, 0.0472116209, 0.1726513979, -0.0744416197],
        [-0.1463270668, 0.0984539476, 0.1354290054, 0.224290114],
        [0.023607999, -0.1144039269, 0.1219468475, -0.1071604884],
      ],
      gp: [
        [0.0497367414, -0.0865942271, -0.0298748789, 0.1706903278],
        [-0.074219701, 0.1141301728, -0.1368954441, 0.0522051437],
        [0.0825101776, -0.1158398113, -0.0819664315, 0.2355536494],
        [0.0184910167, -0.0654363612, -0.1709962489, -0.0783354824],
        [-0.0086925118, 0.031104736, -0.278969704, 0.1976181888],
      ],
      gn: [
        [0.0221733719, -0.1641357466, -0.1095137905, 0.0239693126],
        [0.0871635637, 0.1728804369, 0.0206215077, 0.0457123739],
        [-0.0333064897, 0.0686281904, -0.0906849664, -0.1611120296],
        [0.1278360502, -0.0330175863, 0.0355672434, -0.1459546315],
        [-0.0149154872, 0.083299191, 0.1570228565, -0.0904577004],
      ],
    },
    p1: {
      loss: [2.3374532267, 0.0, 0.0, 1.6924088242, 1.9246592417],
      ga: [
        [-2.0, 2.0, 2.0, -2.0],
        [0.0, -0.0, 0.0, -0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-2.0, 2.0, 0.0, 2.0],
        [2.0, -2.0, 0.0, 0.0],
      ],
      gp: [
        [1.0, -1.0, -1.0, 1.0],
        [-0.0, 0.0, -0.0, 0.0],
        [0.0, -0.0, -0.0, 0.0],
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
      ],
      gn: [
        [1.0, -1.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.0, 0.0, -0.0, -0.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
      ],
    },
    p3: {
      loss: 7.1198959966,
      ga: [
        [-0.0921780016, 1.0714470601, 0.4009773591, -0.9233217305],
        [-0.0327422473, -1.3826247333, 0.6642564348, -0.1622453416],
        [-0.0751407175, 0.0691795687, 0.9251671985, -0.1719933503],
        [-0.5496306094, 0.1691036162, 0.8674948643, 0.8934286393],
        [0.0202840664, -0.4763416212, 0.5412249662, -0.4179323942],
      ],
      gp: [
        [0.0768771788, -0.2330347364, -0.0277367191, 0.9054419385],
        [-0.1990661318, 0.4707170468, -0.6772312148, 0.0984885104],
        [0.1116889544, -0.224351727, -0.6542233478, 1.0271873466],
        [0.0106320409, -0.1331475999, -0.9092184225, -0.1908148213],
        [-0.01286541, 0.2449583592, -1.3634224052, 0.6907933892],
      ],
      gn: [
        [0.0153008228, -0.8384123237, -0.3732406399, 0.017879792],
        [0.2318083791, 0.9119076865, 0.01297478, 0.0637568312],
        [-0.036548237, 0.1551721583, -0.2709438507, -0.8551939962],
        [0.5389985685, -0.0359560163, 0.0417235583, -0.7026138179],
        [-0.0074186564, 0.2313832619, 0.822197439, -0.272860995],
      ],
    },
    pinf: {
      loss: 0.6221412793,
      ga: [
        [-0.0, 0.2, 0.0, -0.2],
        [0.0, -0.0, 0.0, -0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.0, 0.0, 0.2, 0.2],
        [0.0, -0.0, 0.0, 0.0],
      ],
      gp: [
        [0.0, -0.0, -0.0, 0.2],
        [-0.0, 0.0, -0.0, 0.0],
        [0.0, -0.0, -0.0, 0.0],
        [0.0, -0.0, -0.2, -0.0],
        [-0.0, 0.0, -0.2, 0.0],
      ],
      gn: [
        [0.0, -0.2, -0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.0, 0.0, -0.0, -0.0],
        [0.0, -0.0, 0.0, -0.2],
        [-0.0, 0.0, 0.2, -0.0],
      ],
    },
    eps: {
      loss: 0.7544589251,
      ga: [
        [-0.0710587179, 0.2534350754, 0.1414548865, -0.1945076424],
        [0.0, -0.0, 0.0, -0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.144732628, 0.0987006442, 0.1338960488, 0.2243571961],
        [0.0362674247, -0.1597534797, -0.0364283552, -0.0014589694],
      ],
      gp: [
        [0.0477344893, -0.0893730914, -0.0323306402, 0.1693770904],
        [-0.0, 0.0, -0.0, 0.0],
        [0.0, -0.0, -0.0, 0.0],
        [0.0159494567, -0.0665799889, -0.1703816233, -0.0792642565],
        [-0.0293917939, 0.1098742446, -0.1286242461, 0.102564985],
      ],
      gn: [
        [0.0233242286, -0.164061984, -0.1091242463, 0.025130552],
        [0.0, 0.0, 0.0, 0.0],
        [-0.0, 0.0, -0.0, -0.0],
        [0.1287831713, -0.0321206554, 0.0364855745, -0.1450929396],
        [-0.0068756308, 0.0498792352, 0.1650526014, -0.1011060155],
      ],
    },
    one: 1.4131694466,
    plainDefault: 0.0,
  },
};

const f64 = { dtype: "float64" as const };
const opt = (dtype?: "float32" | "float64") => (dtype === undefined ? {} : { dtype });

/** Logical elements of a tensor or GradTensor as a flat number array. */
function values(t: AnyTensor): number[] {
  const inner = GradTensor.isGradTensor(t) ? t.tensor : t;
  const arr = inner.toArray();
  if (Array.isArray(arr)) return (arr as unknown[]).flat(Number.POSITIVE_INFINITY) as number[];
  return [Number(arr)];
}

function flatNums(x: unknown): number[] {
  return Array.isArray(x)
    ? ((x as unknown[]).flat(Number.POSITIVE_INFINITY) as number[])
    : [Number(x)];
}

function expectClose(actual: readonly number[], expected: readonly number[], tol = 1e-9): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] as number;
    const e = expected[i] as number;
    if (Math.abs(a - e) > tol) {
      throw new Error(`element ${i}: expected ${e}, received ${a} (tolerance ${tol})`);
    }
  }
}

const isGrad = (t: AnyTensor): t is GradTensor => GradTensor.isGradTensor(t);

/** Overwrite named parameters of a module with row-major values, keeping dtype and shape. */
function load(module: Module, byName: Record<string, readonly number[]>): void {
  const sd = module.stateDict();
  for (const [name, data] of Object.entries(byName)) {
    const entry = sd.parameters[name];
    if (!entry) throw new Error(`no parameter ${name}`);
    entry.data = [...data];
  }
  module.loadStateDict(sd);
}

function grads(module: Module): Map<string, GradTensor> {
  return new Map(module.namedParameters());
}

// ---------------------------------------------------------------------------
// Training API: automatic gradient tracking
// ---------------------------------------------------------------------------

interface Spec {
  readonly name: string;
  readonly make: () => Module;
  readonly input: () => Tensor;
  /** Whether the layer owns parameters that require grad. */
  readonly trainable: boolean;
  readonly call?: (m: Module, x: AnyTensor) => AnyTensor;
  /** Whether the input can be wrapped in a GradTensor (false for index tensors). */
  readonly gradInput?: boolean;
}

const seq3 = () => randn([2, 5, 4]);
const SPECS: readonly Spec[] = [
  { name: "Linear", make: () => new Linear(3, 2), input: () => randn([4, 3]), trainable: true },
  {
    name: "Conv1d",
    make: () => new Conv1d(2, 3, 2),
    input: () => randn([1, 2, 5]),
    trainable: true,
  },
  {
    name: "Conv2d",
    make: () => new Conv2d(2, 3, 2),
    input: () => randn([1, 2, 4, 4]),
    trainable: true,
  },
  {
    name: "Conv3d",
    make: () => new Conv3d(2, 3, 2),
    input: () => randn([1, 2, 3, 3, 3]),
    trainable: true,
  },
  {
    name: "ConvTranspose1d",
    make: () => new ConvTranspose1d(2, 3, 2),
    input: () => randn([1, 2, 5]),
    trainable: true,
  },
  {
    name: "ConvTranspose2d",
    make: () => new ConvTranspose2d(2, 3, 2),
    input: () => randn([1, 2, 4, 4]),
    trainable: true,
  },
  {
    name: "BatchNorm1d",
    make: () => new BatchNorm1d(3),
    input: () => randn([4, 3]),
    trainable: true,
  },
  {
    name: "BatchNorm2d",
    make: () => new BatchNorm2d(2),
    input: () => randn([2, 2, 3, 3]),
    trainable: true,
  },
  { name: "LayerNorm", make: () => new LayerNorm(3), input: () => randn([4, 3]), trainable: true },
  {
    name: "GroupNorm",
    make: () => new GroupNorm(1, 2),
    input: () => randn([2, 2, 3]),
    trainable: true,
  },
  {
    name: "InstanceNorm1d",
    make: () => new InstanceNorm1d(2),
    input: () => randn([2, 2, 3]),
    trainable: true,
  },
  { name: "RMSNorm", make: () => new RMSNorm(3), input: () => randn([4, 3]), trainable: true },
  { name: "PReLU", make: () => new PReLU(), input: () => randn([4, 3]), trainable: true },
  { name: "RNN", make: () => new RNN(3, 4), input: () => randn([2, 5, 3]), trainable: true },
  { name: "LSTM", make: () => new LSTM(3, 4), input: () => randn([2, 5, 3]), trainable: true },
  { name: "GRU", make: () => new GRU(3, 4), input: () => randn([2, 5, 3]), trainable: true },
  {
    name: "MultiheadAttention",
    make: () => new MultiheadAttention(4, 2),
    input: seq3,
    trainable: true,
    call: (m, x) => (m as MultiheadAttention).forward(x, x, x),
  },
  {
    name: "TransformerEncoderLayer",
    make: () => new TransformerEncoderLayer(4, 2, 8),
    input: seq3,
    trainable: true,
  },
  {
    name: "TransformerDecoderLayer",
    make: () => new TransformerDecoderLayer(4, 2, 8),
    input: seq3,
    trainable: true,
    call: (m, x) => (m as TransformerDecoderLayer).forward(x, x),
  },
  {
    name: "FullTransformer",
    make: () => new FullTransformer(4, 2, 1, 1, 8),
    input: seq3,
    trainable: true,
    call: (m, x) => (m as FullTransformer).forward(x, x),
  },
  {
    name: "SpectralNorm",
    make: () => new SpectralNorm(new Linear(3, 2)),
    input: () => randn([4, 3]),
    trainable: true,
  },
  {
    name: "Sequential",
    make: () => new Sequential(new Linear(3, 4), new ReLU(), new Linear(4, 2)),
    input: () => randn([4, 3]),
    trainable: true,
  },
  {
    name: "Embedding",
    make: () => new Embedding(6, 3),
    input: () => tensor([0, 2, 5, 2], { dtype: "int32" }),
    trainable: true,
    gradInput: false,
  },
  // Parameter-free layers
  { name: "ReLU", make: () => new ReLU(), input: () => randn([4, 3]), trainable: false },
  { name: "Softmax", make: () => new Softmax(), input: () => randn([4, 3]), trainable: false },
  { name: "GLU", make: () => new GLU(), input: () => randn([4, 4]), trainable: false },
  { name: "Dropout", make: () => new Dropout(0.5), input: () => randn([4, 3]), trainable: false },
  {
    name: "Dropout2d",
    make: () => new Dropout2d(0.5),
    input: () => randn([1, 2, 2, 2]),
    trainable: false,
  },
  {
    name: "AlphaDropout",
    make: () => new AlphaDropout(0.5),
    input: () => randn([4, 3]),
    trainable: false,
  },
  {
    name: "MaxPool2d",
    make: () => new MaxPool2d(2),
    input: () => randn([1, 2, 4, 4]),
    trainable: false,
  },
  {
    name: "AvgPool2d",
    make: () => new AvgPool2d(2),
    input: () => randn([1, 2, 4, 4]),
    trainable: false,
  },
  {
    name: "AdaptiveAvgPool2d",
    make: () => new AdaptiveAvgPool2d(2),
    input: () => randn([1, 2, 4, 4]),
    trainable: false,
  },
  {
    name: "ZeroPad2d",
    make: () => new ZeroPad2d(1),
    input: () => randn([1, 1, 2, 2]),
    trainable: false,
  },
  {
    name: "Upsample",
    make: () => new Upsample({ scaleFactor: 2 }),
    input: () => randn([1, 1, 2, 2]),
    trainable: false,
  },
  { name: "Flatten", make: () => new Flatten(), input: () => randn([2, 3, 2]), trainable: false },
  {
    name: "LocalResponseNorm",
    make: () => new LocalResponseNorm(2),
    input: () => randn([2, 3, 3]),
    trainable: false,
  },
  {
    name: "PositionalEncoding",
    make: () => new PositionalEncoding(4),
    input: seq3,
    trainable: false,
  },
  {
    name: "Identity",
    make: () => new Identity(),
    input: () => randn([2, 3]),
    trainable: false,
    gradInput: false,
  },
];

const run = (spec: Spec, m: Module, x: AnyTensor): AnyTensor =>
  spec.call ? spec.call(m, x) : m.forward(x);

describe("training API: a plain tensor input tracks the weights", () => {
  for (const spec of SPECS) {
    it(`${spec.name}: tracking follows the weights, noGrad and frozen weights give plain tensors`, () => {
      const m = spec.make();
      const x = spec.input();
      const out = run(spec, m, x);
      if (spec.trainable) {
        expect(isGrad(out)).toBe(true);
        expect((out as GradTensor).requiresGrad).toBe(true);
        (out as GradTensor).sum().backward();
        let withGrad = 0;
        for (const p of m.parameters()) if (p.grad !== null) withGrad++;
        expect(withGrad).toBeGreaterThan(0);
        m.zeroGrad();
      } else {
        expect(isGrad(out)).toBe(false);
      }

      // noGrad(): plain tensor, same shape
      const plain = noGrad(() => run(spec, spec.make(), x));
      expect(isGrad(plain)).toBe(false);

      // every parameter frozen: plain tensor
      const frozen = spec.make();
      frozen.freezeParameters();
      expect(isGrad(run(spec, frozen, x))).toBe(false);
    });
  }

  for (const spec of SPECS.filter((s) => s.gradInput !== false)) {
    it(`${spec.name}: a GradTensor input always gives a GradTensor and receives gradients`, () => {
      const m = spec.make();
      const raw = spec.input();
      const x = GradTensor.fromTensor(raw, { requiresGrad: true });
      const out = run(spec, m, x);
      expect(isGrad(out)).toBe(true);
      (out as GradTensor).sum().backward();
      expect(x.grad).not.toBeNull();

      // inside noGrad the result is still a GradTensor, but it records no graph
      const quiet = noGrad(() => run(spec, spec.make(), x));
      expect(isGrad(quiet)).toBe(true);
      expect((quiet as GradTensor).requiresGrad).toBe(false);
    });
  }

  it("Identity returns its input object unchanged", () => {
    const x = randn([2, 2]);
    expect(new Identity().forward(x)).toBe(x);
    const g = GradTensor.fromTensor(x, { requiresGrad: true });
    expect(new Identity().forward(g)).toBe(g);
  });

  it("eval() alone does not disable tracking (PyTorch)", () => {
    const model = new Sequential(new Linear(3, 2), new Dropout(0.3));
    model.eval();
    expect(isGrad(model.forward(randn([4, 3])))).toBe(true);
    expect(isGrad(noGrad(() => model.forward(randn([4, 3]))))).toBe(false);
  });

  it("unfreezing restores tracking", () => {
    const layer = new Linear(3, 2);
    layer.freezeParameters();
    expect(isGrad(layer.forward(randn([2, 3])))).toBe(false);
    layer.unfreezeParameters(["bias"]);
    expect(isGrad(layer.forward(randn([2, 3])))).toBe(true);
  });

  it("the data itself is not tracked: only the weights receive gradients", () => {
    const layer = new Linear(3, 2, f64);
    const x = tensor([[1, 2, 3]], f64);
    const out = layer.forward(x);
    expect(isGrad(out)).toBe(true);
    (out as GradTensor).sum().backward();
    const g = grads(layer);
    expect(g.get("weight")?.grad).not.toBeNull();
    expect(g.get("bias")?.grad).not.toBeNull();
  });

  it("a plain tensor into a layer without weights stays plain, also after trainable layers are frozen", () => {
    const seq = new Sequential(new ReLU(), new Flatten());
    expect(isGrad(seq.forward(randn([2, 3, 2])))).toBe(false);
  });

  it("matches PyTorch: Linear + mseLoss gradients without wrapping the data", () => {
    const ref = REF.linear;
    const layer = new Linear(3, 2, f64);
    load(layer, { weight: flatNums(ref.W), bias: flatNums(ref.b) });
    const pred = layer.forward(tensor(ref.x, f64));
    expect(isGrad(pred)).toBe(true);
    expectClose(values(pred), flatNums(ref.pred), 1e-9);
    const loss = mseLoss(pred as GradTensor, tensor(ref.y, f64));
    expect(values(loss)[0]).toBeCloseTo(ref.loss, 9);
    loss.backward();
    const g = grads(layer);
    expectClose(values(g.get("weight")?.grad as Tensor), flatNums(ref.gW), 1e-9);
    expectClose(values(g.get("bias")?.grad as Tensor), flatNums(ref.gb), 1e-9);
  });

  it("a training loop works with plain tensors: forward, loss, backward, step", () => {
    setSeed(5);
    const X = randn([64, 3]);
    const y = dot(X, tensor([[1.5], [-2], [0.5]]));
    const model = new Sequential(new Linear(3, 8), new ReLU(), new Linear(8, 1));
    const optimizer = new Adam(model.parameters(), { lr: 0.05 });
    const before = values(model.getLayer(0).forward(X)).slice(0, 4);
    let first = Number.NaN;
    let last = Number.NaN;
    for (let i = 0; i < 150; i++) {
      optimizer.zeroGrad();
      const pred = model.forward(X);
      expect(isGrad(pred)).toBe(true);
      const loss = mseLoss(pred as GradTensor, y);
      loss.backward();
      optimizer.step();
      const v = values(loss)[0] as number;
      if (i === 0) first = v;
      last = v;
    }
    expect(last).toBeLessThan(first * 0.05);
    const after = values(model.getLayer(0).forward(X)).slice(0, 4);
    expect(after).not.toEqual(before);
    // evaluation without a graph
    const evalOut = noGrad(() => model.forward(X));
    expect(isGrad(evalOut)).toBe(false);
  });

  it("forward hooks of call() see the tracked output", () => {
    const seq = new Sequential(new Linear(3, 2));
    let seen: boolean | undefined;
    seq.registerForwardHook((_m, _inputs, output) => {
      seen = isGrad(output);
      return undefined;
    });
    seq.call(randn([2, 3]));
    expect(seen).toBe(true);
  });

  it("multi-input layers: any GradTensor that requires grad makes the result tracked", () => {
    const mha = new MultiheadAttention(4, 2);
    mha.freezeParameters();
    const q = randn([1, 3, 4]);
    expect(isGrad(mha.forward(q, q, q))).toBe(false);
    const k = GradTensor.fromTensor(randn([1, 3, 4]), { requiresGrad: true });
    const out = mha.forward(q, k, k);
    expect(isGrad(out)).toBe(true);
    (out as GradTensor).sum().backward();
    expect(k.grad).not.toBeNull();
  });

  it("recurrent forwardWithState and EmbeddingBag follow the same rule", () => {
    const lstm = new LSTM(3, 4);
    const x = randn([2, 5, 3]);
    const [out, [h, c]] = lstm.forwardWithState(x);
    expect([isGrad(out), isGrad(h), isGrad(c)]).toEqual([true, true, true]);
    const [out2, [h2, c2]] = noGrad(() => lstm.forwardWithState(x));
    expect([isGrad(out2), isGrad(h2), isGrad(c2)]).toEqual([false, false, false]);
    const [gout, gh] = new GRU(3, 4).forwardWithState(x);
    expect([isGrad(gout), isGrad(gh)]).toEqual([true, true]);
    const [rout, rh] = noGrad(() => new RNN(3, 4).forwardWithState(x));
    expect([isGrad(rout), isGrad(rh)]).toEqual([false, false]);

    const bag = new EmbeddingBag(6, 3, { mode: "sum" });
    const bagOut = bag.forward(
      tensor([0, 1, 2, 3], { dtype: "int32" }),
      tensor([0, 2], { dtype: "int32" })
    );
    expect(isGrad(bagOut)).toBe(true);
    expect(
      isGrad(
        noGrad(() =>
          bag.forward(tensor([0, 1], { dtype: "int32" }), tensor([0], { dtype: "int32" }))
        )
      )
    ).toBe(false);
  });

  it("BatchNorm without affine parameters returns a plain tensor but still updates its statistics", () => {
    const bn = new BatchNorm1d(2, { affine: false });
    const x = tensor([
      [1, 2],
      [3, 6],
    ]);
    const out = bn.forward(x);
    expect(isGrad(out)).toBe(false);
    const sd = bn.stateDict();
    expectClose(sd.buffers.running_mean?.data as number[], [0.2, 0.4], 1e-6);
  });

  it("Dropout in training mode keeps a plain input plain", () => {
    const d = new Dropout(0.5);
    const out = d.forward(randn([8, 8]));
    expect(isGrad(out)).toBe(false);
    expect(out.shape).toEqual([8, 8]);
  });
});

// ---------------------------------------------------------------------------
// Layer dtype rule
// ---------------------------------------------------------------------------

type DT = "float32" | "float64";

const PARAM_LAYERS: ReadonlyArray<readonly [string, (d?: DT) => Module, number[]]> = [
  ["Linear", (d) => new Linear(3, 2, opt(d)), [4, 3]],
  ["Conv1d", (d) => new Conv1d(2, 3, 2, opt(d)), [1, 2, 5]],
  ["Conv2d", (d) => new Conv2d(2, 3, 2, opt(d)), [1, 2, 4, 4]],
  ["Conv3d", (d) => new Conv3d(2, 3, 2, opt(d)), [1, 2, 3, 3, 3]],
  ["ConvTranspose1d", (d) => new ConvTranspose1d(2, 3, 2, opt(d)), [1, 2, 5]],
  ["ConvTranspose2d", (d) => new ConvTranspose2d(2, 3, 2, opt(d)), [1, 2, 4, 4]],
  ["BatchNorm1d", (d) => new BatchNorm1d(3, opt(d)), [4, 3]],
  ["BatchNorm2d", (d) => new BatchNorm2d(2, opt(d)), [2, 2, 3, 3]],
  ["LayerNorm", (d) => new LayerNorm(3, opt(d)), [4, 3]],
  ["GroupNorm", (d) => new GroupNorm(1, 2, opt(d)), [2, 2, 3]],
  ["RMSNorm", (d) => new RMSNorm(3, opt(d)), [4, 3]],
  ["PReLU", (d) => new PReLU(1, 0.25, opt(d)), [4, 3]],
  ["RNN", (d) => new RNN(3, 4, opt(d)), [2, 5, 3]],
  ["LSTM", (d) => new LSTM(3, 4, opt(d)), [2, 5, 3]],
  ["GRU", (d) => new GRU(3, 4, opt(d)), [2, 5, 3]],
  ["MultiheadAttention", (d) => new MultiheadAttention(4, 2, opt(d)), [2, 5, 4]],
  ["TransformerEncoderLayer", (d) => new TransformerEncoderLayer(4, 2, 8, opt(d)), [2, 5, 4]],
];

function forwardOne(m: Module, x: AnyTensor): AnyTensor {
  return m instanceof MultiheadAttention ? m.forward(x, x, x) : m.forward(x);
}

describe("layer dtype rule: layers with parameters compute in the parameter dtype", () => {
  for (const [name, make, shape] of PARAM_LAYERS) {
    it(`${name}: float64 and integer inputs are cast to float32 by default`, () => {
      const m = make();
      for (const p of m.parameters()) expect(p.dtype).toBe("float32");
      const x64 = randn(shape, f64);
      expect(forwardOne(m, x64).dtype).toBe("float32");
      const xi = randn(shape).astype("int32");
      expect(forwardOne(m, xi).dtype).toBe("float32");
      // GradTensor inputs follow the same rule, differentiably
      const g = GradTensor.fromTensor(randn(shape, f64), { requiresGrad: true });
      const out = forwardOne(m, g) as GradTensor;
      expect(out.dtype).toBe("float32");
      out.sum().backward();
      expect(g.grad?.dtype).toBe("float64");
    });

    it(`${name}: a float64 layer computes in float64, also for float32 input`, () => {
      const m = make("float64");
      for (const p of m.parameters()) expect(p.dtype).toBe("float64");
      expect(forwardOne(m, randn(shape)).dtype).toBe("float64");
      expect(forwardOne(m, randn(shape, f64)).dtype).toBe("float64");
    });
  }

  it("Linear accepts boolean input", () => {
    const layer = new Linear(2, 1, f64);
    load(layer, { weight: [1, 2], bias: [0.5] });
    const out = layer.forward(tensor([[1, 0]], { dtype: "bool" }));
    expectClose(values(out), [1.5]);
    expect(out.dtype).toBe("float64");
  });

  it("recurrent layers accept integer input and integer initial states", () => {
    const rnn = new RNN(2, 3);
    const x = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { dtype: "int32" }
    );
    expect(rnn.forward(x).dtype).toBe("float32");
    const lstm = new LSTM(2, 3);
    const h0 = tensor([[0, 0, 0]], { dtype: "int32" });
    const [out] = lstm.forwardWithState(x, h0, h0);
    expect(out.dtype).toBe("float32");
    expect(() => rnn.forward(tensor([["a", "b"]]))).toThrow(/string dtype/);
  });

  it("a float32 layer on float64 input still matches PyTorch to float32 precision", () => {
    const ref = REF.layernorm;
    const x = tensor(ref.x, f64);
    const f32Out = new LayerNorm(4).forward(x);
    expect(f32Out.dtype).toBe("float32");
    expectClose(values(f32Out), flatNums(ref.out), 2e-6);
    const f64Out = new LayerNorm(4, f64).forward(x);
    expectClose(values(f64Out), flatNums(ref.out), 1e-9);
  });

  it("Conv2d (float64 layer) matches PyTorch outputs and gradients", () => {
    const ref = REF.conv2d;
    const conv = new Conv2d(2, 3, 2, f64);
    load(conv, { weight: flatNums(ref.W), bias: flatNums(ref.b) });
    const x = GradTensor.fromTensor(tensor(ref.x, f64), { requiresGrad: true });
    const out = conv.forward(x);
    expectClose(values(out), flatNums(ref.out), 1e-9);
    const weighted = out.mul(GradTensor.fromTensor(tensor(ref.G, f64)));
    weighted.sum().backward();
    const g = grads(conv);
    expectClose(values(g.get("weight")?.grad as Tensor), flatNums(ref.gW), 1e-9);
    expectClose(values(g.get("bias")?.grad as Tensor), flatNums(ref.gb), 1e-9);
    expectClose(values(x.grad as Tensor), flatNums(ref.gx), 1e-9);
  });

  it("the global default dtype decides the dtype of layers built without an option", () => {
    // Linear uses the global default (float32 unless changed)
    expect(new Linear(2, 2).getWeight().dtype).toBe("float32");
    expect(new Linear(2, 2, { dtype: "float64" }).getWeight().dtype).toBe("float64");
    expect(() => new Linear(2, 2, { dtype: "int32" as unknown as DT })).toThrow(
      InvalidParameterError
    );
    expect(() => new Conv2d(1, 1, 2, { dtype: "float16" as unknown as DT })).toThrow(
      InvalidParameterError
    );
  });
});

const KEEP_FLOAT: ReadonlyArray<readonly [string, () => Module, number[]]> = [
  ["ReLU", () => new ReLU(), [3, 4]],
  ["Sigmoid", () => new Sigmoid(), [3, 4]],
  ["Tanh", () => new Tanh(), [3, 4]],
  ["ELU", () => new ELU(), [3, 4]],
  ["GELU", () => new GELU(), [3, 4]],
  ["Softmax", () => new Softmax(), [3, 4]],
  ["LogSoftmax", () => new LogSoftmax(), [3, 4]],
  ["Softmin", () => new Softmin(), [3, 4]],
  ["Softplus", () => new Softplus(), [3, 4]],
  ["Softplus(beta)", () => new Softplus(2), [3, 4]],
  ["Mish", () => new Mish(), [3, 4]],
  ["SELU", () => new SELU(), [3, 4]],
  ["Hardswish", () => new Hardswish(), [3, 4]],
  ["Softsign", () => new Softsign(), [3, 4]],
  ["GLU", () => new GLU(), [3, 4]],
  ["Dropout", () => new Dropout(0.5), [3, 4]],
  ["Dropout2d", () => new Dropout2d(0.5), [1, 2, 2, 2]],
  ["AlphaDropout", () => new AlphaDropout(0.5), [3, 4]],
  ["MaxPool1d", () => new MaxPool1d(2), [1, 2, 4]],
  ["AvgPool1d", () => new AvgPool1d(2), [1, 2, 4]],
  ["MaxPool2d", () => new MaxPool2d(2), [1, 2, 4, 4]],
  ["AvgPool2d", () => new AvgPool2d(2), [1, 2, 4, 4]],
  [
    "AvgPool2d(countIncludePad false)",
    () => new AvgPool2d(2, { countIncludePad: false }),
    [1, 2, 4, 4],
  ],
  ["MaxPool3d", () => new MaxPool3d(2), [1, 2, 4, 4, 4]],
  ["AvgPool3d", () => new AvgPool3d(2), [1, 2, 4, 4, 4]],
  ["AdaptiveAvgPool2d", () => new AdaptiveAvgPool2d(2), [1, 2, 4, 4]],
  ["ZeroPad2d", () => new ZeroPad2d(1), [1, 1, 2, 2]],
  ["ReflectionPad2d", () => new ReflectionPad2d(1), [1, 1, 3, 3]],
  ["Upsample", () => new Upsample({ scaleFactor: 2 }), [1, 1, 2, 2]],
  ["Flatten", () => new Flatten(), [2, 3, 2]],
  ["Unflatten", () => new Unflatten(1, [3, 1]), [2, 3]],
  ["LocalResponseNorm", () => new LocalResponseNorm(2), [2, 3, 3]],
  ["PositionalEncoding", () => new PositionalEncoding(4), [2, 5, 4]],
  ["Identity", () => new Identity(), [2, 3]],
];

describe("layer dtype rule: layers without parameters keep the input float dtype", () => {
  for (const [name, make, shape] of KEEP_FLOAT) {
    it(`${name}`, () => {
      expect(make().forward(randn(shape)).dtype).toBe("float32");
      expect(make().forward(randn(shape, f64)).dtype).toBe("float64");
      const g = GradTensor.fromTensor(randn(shape, f64), { requiresGrad: true });
      expect(make().forward(g).dtype).toBe("float64");
    });
  }

  it("activation values on float32 input match PyTorch float32 results", () => {
    const ref = REF.act;
    const x = tensor(ref.x, { dtype: "float32" });
    const cases: Array<[string, Module, readonly (readonly number[])[]]> = [
      ["elu", new ELU(), ref.elu],
      ["gelu", new GELU(), ref.gelu],
      ["softmax", new Softmax(), ref.softmax],
      ["logsoftmax", new LogSoftmax(), ref.logsoftmax],
      ["softplus", new Softplus(), ref.softplus],
      ["tanh", new Tanh(), ref.tanh],
      ["selu", new SELU(), ref.selu],
      ["hardswish", new Hardswish(), ref.hardswish],
      ["softsign", new Softsign(), ref.softsign],
      ["sigmoid", new Sigmoid(), ref.sigmoid],
      ["mish", new Mish(), ref.mish],
      ["softmin", new Softmin(), ref.softmin],
    ];
    for (const [name, layer, expected] of cases) {
      const out = layer.forward(x);
      expect(out.dtype, name).toBe("float32");
      expectClose(values(out), flatNums(expected), 3e-6);
    }
  });

  it("non-contiguous inputs give the same result as their contiguous copy", () => {
    const base = randn([4, 3]);
    const view = transpose(base); // shape [3, 4], strided
    const copy = tensor(view.toArray() as number[][]);
    for (const layer of [new ELU(), new Softmax(-1), new Softsign(), new Hardswish(), new Tanh()]) {
      expectClose(values(layer.forward(view)), values(layer.forward(copy)), 1e-7);
    }
    const lin = new Linear(4, 2);
    expectClose(values(lin.forward(view)), values(lin.forward(copy)), 1e-6);
  });

  it("negative axes behave like their positive equivalents", () => {
    const x = randn([2, 3, 4]);
    expectClose(values(new Softmax(-1).forward(x)), values(new Softmax(2).forward(x)), 1e-7);
    expectClose(values(new LogSoftmax(-2).forward(x)), values(new LogSoftmax(1).forward(x)), 1e-6);
  });
});

// ---------------------------------------------------------------------------
// Weight initialization (PyTorch defaults)
// ---------------------------------------------------------------------------

function uniformStats(data: readonly number[], bound: number): void {
  let max = 0;
  let sum = 0;
  for (const v of data) {
    max = Math.max(max, Math.abs(v));
    sum += v;
  }
  const mean = sum / data.length;
  let sq = 0;
  for (const v of data) sq += (v - mean) ** 2;
  const std = Math.sqrt(sq / data.length);
  expect(max).toBeLessThanOrEqual(bound * (1 + 1e-6));
  expect(max).toBeGreaterThan(bound * 0.95);
  expect(Math.abs(mean)).toBeLessThan(bound * 0.05);
  // U(-b, b) has standard deviation b / sqrt(3)
  expect(Math.abs(std / (bound / Math.sqrt(3)) - 1)).toBeLessThan(0.05);
}

/** Small samples (biases): inside the bound, not all zero, and not a constant. */
function withinBound(data: readonly number[], bound: number): void {
  expect(data.every((v) => Math.abs(v) <= bound * (1 + 1e-6))).toBe(true);
  expect(new Set(data).size).toBeGreaterThan(1);
}

describe("weight initialization matches PyTorch", () => {
  it("Linear: weight and bias are U(-1/sqrt(in), 1/sqrt(in))", () => {
    setSeed(1);
    const layer = new Linear(100, 200);
    const bound = 1 / Math.sqrt(100);
    uniformStats(values(layer.getWeight()), bound);
    const bias = layer.getBias() as Tensor;
    uniformStats(values(bias), bound);
    expect(values(bias).some((v) => v !== 0)).toBe(true);
  });

  it("Linear without bias, float64 layer", () => {
    const layer = new Linear(50, 80, { bias: false, dtype: "float64" });
    expect(layer.getBias()).toBeUndefined();
    uniformStats(values(layer.getWeight()), 1 / Math.sqrt(50));
  });

  it("Conv layers: bound 1/sqrt(inChannels * kernel volume)", () => {
    setSeed(2);
    const c1 = new Conv1d(16, 40, 5);
    uniformStats(values(c1.weight), 1 / Math.sqrt(16 * 5));
    withinBound(values(c1.bias as GradTensor), 1 / Math.sqrt(16 * 5));
    const c2 = new Conv2d(8, 64, [3, 5]);
    uniformStats(values(c2.weight), 1 / Math.sqrt(8 * 15));
    withinBound(values(c2.bias as GradTensor), 1 / Math.sqrt(8 * 15));
    const c3 = new Conv3d(4, 32, 3);
    uniformStats(values(c3.weight), 1 / Math.sqrt(4 * 27));
    withinBound(values(c3.bias as GradTensor), 1 / Math.sqrt(4 * 27));
  });

  it("ConvTranspose layers: fan in is outChannels * kernel volume, as in PyTorch", () => {
    setSeed(3);
    const t1 = new ConvTranspose1d(16, 40, 5);
    uniformStats(values(t1.weight), 1 / Math.sqrt(40 * 5));
    withinBound(values(t1.bias as GradTensor), 1 / Math.sqrt(40 * 5));
    const t2 = new ConvTranspose2d(8, 64, 3);
    uniformStats(values(t2.weight), 1 / Math.sqrt(64 * 9));
    withinBound(values(t2.bias as GradTensor), 1 / Math.sqrt(64 * 9));
  });

  it("RNN, LSTM and GRU: every weight and bias is U(-1/sqrt(hidden), 1/sqrt(hidden))", () => {
    setSeed(4);
    const hidden = 40;
    const bound = 1 / Math.sqrt(hidden);
    for (const layer of [
      new RNN(30, hidden, { numLayers: 2, bidirectional: true }),
      new LSTM(30, hidden, { numLayers: 2 }),
      new GRU(30, hidden, { numLayers: 2, bidirectional: true }),
    ]) {
      const all: number[] = [];
      const biases: number[] = [];
      for (const [name, p] of layer.namedParameters()) {
        const v = values(p);
        for (const x of v) all.push(x);
        if (name.startsWith("bias")) for (const x of v) biases.push(x);
      }
      uniformStats(all, bound);
      uniformStats(biases, bound);
      expect(biases.some((v) => v !== 0)).toBe(true);
    }
  });

  it("initialization is reproducible with the global seed", () => {
    setSeed(11);
    const a = new Linear(5, 4);
    const ca = new Conv2d(2, 3, 2);
    const ra = new GRU(3, 4);
    setSeed(11);
    const b = new Linear(5, 4);
    const cb = new Conv2d(2, 3, 2);
    const rb = new GRU(3, 4);
    expect(values(a.getWeight())).toEqual(values(b.getWeight()));
    expect(values(ca.weight)).toEqual(values(cb.weight));
    expect(values(grads(ra).get("weight_ih_l0") as GradTensor)).toEqual(
      values(grads(rb).get("weight_ih_l0") as GradTensor)
    );
    setSeed(12);
    const c = new Linear(5, 4);
    expect(values(c.getWeight())).not.toEqual(values(a.getWeight()));
  });
});

// ---------------------------------------------------------------------------
// Embedding paddingIdx
// ---------------------------------------------------------------------------

describe("Embedding paddingIdx follows PyTorch", () => {
  const table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

  it("starts the padding row at zero", () => {
    setSeed(1);
    const emb = new Embedding(5, 3, { paddingIdx: 2 });
    expect(values(emb.weight).slice(6, 9)).toEqual([0, 0, 0]);
    expect(
      values(emb.weight)
        .slice(0, 6)
        .some((v) => v !== 0)
    ).toBe(true);
    const neg = new Embedding(5, 3, { paddingIdx: -1 });
    expect(neg.paddingIdx).toBe(4);
    expect(values(neg.weight).slice(12, 15)).toEqual([0, 0, 0]);
  });

  it("returns the stored row and blocks only its gradient (values and gradients match PyTorch)", () => {
    const ref = REF.embedding;
    const emb = new Embedding(5, 3, { paddingIdx: 2, dtype: "float64" });
    load(emb, { weight: table });
    const out = emb.forward(tensor(ref.idx, { dtype: "int32" }));
    expect(isGrad(out)).toBe(true);
    expectClose(values(out), flatNums(ref.out));
    // index 2 is the padding index: its stored row (7, 8, 9) is returned
    expect(values(out).slice(3, 9)).toEqual([7, 8, 9, 7, 8, 9]);
    (out as GradTensor)
      .mul(GradTensor.fromTensor(tensor(ref.G, f64)))
      .sum()
      .backward();
    expectClose(values(emb.weight.grad as Tensor), flatNums(ref.grad));
    expect(values(emb.weight.grad as Tensor).slice(6, 9)).toEqual([0, 0, 0]);
  });

  it("returns whatever was written into the padding row", () => {
    const emb = new Embedding(4, 2, { paddingIdx: 1, dtype: "float64" });
    load(emb, { weight: [1, 1, 5, 6, 2, 2, 3, 3] });
    expectClose(values(emb.forward(tensor([1, 0], { dtype: "int32" }))), [5, 6, 1, 1]);
  });

  it("handles index tensors of any rank", () => {
    const ref = REF.embedding2d;
    const emb = new Embedding(5, 3, { paddingIdx: 2, dtype: "float64" });
    load(emb, { weight: table });
    const out = emb.forward(tensor(ref.idx, { dtype: "int64" }));
    expect(out.shape).toEqual([2, 2, 3]);
    expectClose(values(out), flatNums(ref.out));
    // 0-d index
    expect(emb.forward(tensor(3, { dtype: "int32" })).shape).toEqual([3]);
    // empty index
    expect(emb.forward(tensor([], { dtype: "int32" })).shape).toEqual([0, 3]);
  });

  it("fromPretrained keeps the given padding row and returns it", () => {
    const emb = Embedding.fromPretrained(
      tensor(
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        f64
      ),
      {
        paddingIdx: 1,
        freeze: false,
        dtype: "float64",
      }
    );
    expect(emb.weight.dtype).toBe("float64");
    expectClose(values(emb.forward(tensor([1], { dtype: "int32" }))), [3, 4]);
    (emb.forward(tensor([1, 2], { dtype: "int32" })) as GradTensor).sum().backward();
    expectClose(values(emb.weight.grad as Tensor), [0, 0, 0, 0, 1, 1]);
  });

  it("without paddingIdx every row gets a gradient; EmbeddingBag still skips the padding index", () => {
    const emb = new Embedding(3, 2, { dtype: "float64" });
    (emb.forward(tensor([0, 1, 1], { dtype: "int32" })) as GradTensor).sum().backward();
    expectClose(values(emb.weight.grad as Tensor), [1, 1, 2, 2, 0, 0]);
    const bag = new EmbeddingBag(3, 2, { mode: "sum", paddingIdx: 1, dtype: "float64" });
    load(bag, { weight: [1, 1, 100, 100, 3, 3] });
    const out = bag.forward(tensor([[0, 1, 2]], { dtype: "int32" }));
    expectClose(values(out), [4, 4]);
  });

  it("validates indices", () => {
    const emb = new Embedding(3, 2);
    expect(() => emb.forward(tensor([3], { dtype: "int32" }))).toThrow(InvalidParameterError);
    expect(() => emb.forward(tensor([-1], { dtype: "int32" }))).toThrow(InvalidParameterError);
    expect(() => emb.forward(tensor([0.5]))).toThrow(InvalidParameterError);
    expect(() => new Embedding(3, 2, { paddingIdx: 3 })).toThrow(InvalidParameterError);
  });
});

// ---------------------------------------------------------------------------
// Convolution kernels that do not fit
// ---------------------------------------------------------------------------

describe("convolution layers throw ShapeError when the kernel does not fit", () => {
  it("Conv1d / Conv2d / Conv3d", () => {
    expect(() => new Conv1d(1, 1, 5).forward(randn([1, 1, 3]))).toThrow(ShapeError);
    expect(() => new Conv2d(1, 1, 5).forward(randn([1, 1, 3, 3]))).toThrow(ShapeError);
    expect(() => new Conv2d(1, 1, [2, 5]).forward(randn([1, 1, 4, 3]))).toThrow(ShapeError);
    expect(() => new Conv3d(1, 1, 5).forward(randn([1, 1, 3, 3, 3]))).toThrow(ShapeError);
    expect(() => new Conv2d(1, 1, 5).forward(GradTensor.fromTensor(randn([1, 1, 3, 3])))).toThrow(
      ShapeError
    );
  });

  it("the error is not an InvalidParameterError", () => {
    try {
      new Conv2d(1, 1, 5).forward(randn([1, 1, 3, 3]));
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(ShapeError);
      expect(e).not.toBeInstanceOf(InvalidParameterError);
    }
  });

  it("padding that makes the kernel fit is accepted", () => {
    expect(new Conv1d(1, 1, 5, { padding: 1 }).forward(randn([1, 1, 3])).shape).toEqual([1, 1, 1]);
    expect(new Conv2d(1, 1, 5, { padding: 1 }).forward(randn([1, 1, 3, 3])).shape).toEqual([
      1, 1, 1, 1,
    ]);
    expect(new Conv2d(1, 1, 3).forward(randn([1, 1, 3, 3])).shape).toEqual([1, 1, 1, 1]);
  });

  it("transposed convolutions and pooling already used ShapeError", () => {
    expect(() => new ConvTranspose1d(1, 1, 3, { padding: 5 }).forward(randn([1, 1, 3]))).toThrow(
      ShapeError
    );
    expect(() => new ConvTranspose2d(1, 1, 3, { padding: 5 }).forward(randn([1, 1, 3, 3]))).toThrow(
      ShapeError
    );
    expect(() => new MaxPool2d(5).forward(randn([1, 1, 3, 3]))).toThrow(ShapeError);
    expect(() => new AvgPool2d(5).forward(randn([1, 1, 3, 3]))).toThrow(ShapeError);
    expect(() => new MaxPool1d(5).forward(randn([1, 1, 3]))).toThrow(ShapeError);
  });
});

// ---------------------------------------------------------------------------
// tripletMarginLoss
// ---------------------------------------------------------------------------

describe("tripletMarginLoss", () => {
  const T = REF.triplet;
  const triple = (requiresGrad: boolean) => {
    const mk = (data: number[][]) => GradTensor.fromTensor(tensor(data, f64), { requiresGrad });
    return [mk(T.a), mk(T.p), mk(T.n)] as const;
  };
  const plainTriple = () => [tensor(T.a, f64), tensor(T.p, f64), tensor(T.n, f64)] as const;

  const cases = [
    ["default eps, margin 2.5", { margin: 2.5 }, T.default],
    ["swap", { swap: true, margin: 3 }, T.swap],
    ["p = 1, no reduction", { p: 1, margin: 4, reduction: "none" as const }, T.p1],
    ["p = 3, sum, swap", { p: 3, margin: 3, reduction: "sum" as const, swap: true }, T.p3],
    ["p = Infinity", { p: Number.POSITIVE_INFINITY, margin: 2 }, T.pinf],
    ["eps = 1e-2", { eps: 1e-2, margin: 2.5 }, T.eps],
  ] as const;

  for (const [name, options, ref] of cases) {
    it(`${name}: loss and gradients match PyTorch (GradTensor inputs)`, () => {
      const [a, p, n] = triple(true);
      const loss = tripletMarginLoss(a, p, n, options);
      expect(isGrad(loss)).toBe(true);
      expectClose(values(loss), flatNums(ref.loss), 1e-9);
      loss.sum().backward();
      expectClose(values(a.grad as Tensor), flatNums(ref.ga), 1e-9);
      expectClose(values(p.grad as Tensor), flatNums(ref.gp), 1e-9);
      expectClose(values(n.grad as Tensor), flatNums(ref.gn), 1e-9);
    });

    it(`${name}: plain tensors give the same loss`, () => {
      const [a, p, n] = plainTriple();
      const loss = tripletMarginLoss(a, p, n, options);
      expect(isGrad(loss)).toBe(false);
      expectClose(values(loss), flatNums(ref.loss), 1e-9);
    });
  }

  it("the default margin is 1 and the default reduction is the mean", () => {
    const [a, p, n] = plainTriple();
    expect(values(tripletMarginLoss(a, p, n))[0]).toBeCloseTo(
      flatNums(T.plainDefault)[0] as number,
      10
    );
    expect(tripletMarginLoss(a, p, n).shape).toEqual([]);
  });

  it("the positional margin and reduction still work and equal the options form", () => {
    const [a, p, n] = plainTriple();
    const positional = tripletMarginLoss(a, p, n, 2.5, "sum");
    const named = tripletMarginLoss(a, p, n, { margin: 2.5, reduction: "sum" });
    expect(values(positional)).toEqual(values(named));
    const [ga, gp, gn] = triple(false);
    expectClose(values(tripletMarginLoss(ga, gp, gn, 2.5, "sum")), values(named), 1e-12);
  });

  it("adds eps = 1e-6 to the distance, as PyTorch does", () => {
    const anchor = tensor([[0, 0]], f64);
    const negative = tensor([[3, 4]], f64);
    const loss = tripletMarginLoss(anchor, anchor, negative, { margin: 6 });
    expect(values(loss)[0]).toBeCloseTo(1.0000028142135582, 12);
    const noEps = tripletMarginLoss(anchor, anchor, negative, { margin: 6, eps: 0 });
    expect(values(noEps)[0]).toBeCloseTo(1, 12);
  });

  it("a single triplet (1-D inputs) gives a scalar loss, also without reduction", () => {
    const one = tripletMarginLoss(
      tensor(T.a[0] as number[], f64),
      tensor(T.p[0] as number[], f64),
      tensor(T.n[0] as number[], f64),
      {
        margin: 2.5,
        reduction: "none",
      }
    );
    expect(one.shape).toEqual([]);
    expect(values(one)[0]).toBeCloseTo(flatNums(T.one)[0] as number, 8);
  });

  it("returns float32 for float32 input and keeps float64", () => {
    const f32 = (data: number[][]) => tensor(data, { dtype: "float32" });
    expect(tripletMarginLoss(f32(T.a), f32(T.p), f32(T.n), { margin: 2.5 }).dtype).toBe("float32");
    const [a, p, n] = plainTriple();
    expect(tripletMarginLoss(a, p, n).dtype).toBe("float64");
    const g = GradTensor.fromTensor(f32(T.a), { requiresGrad: true });
    expect(tripletMarginLoss(g, f32(T.p), f32(T.n)).dtype).toBe("float32");
  });

  it("any GradTensor operand makes the loss differentiable, and only it gets a gradient", () => {
    const negative = GradTensor.fromTensor(tensor(T.n, f64), { requiresGrad: true });
    const loss = tripletMarginLoss(tensor(T.a, f64), tensor(T.p, f64), negative, { margin: 2.5 });
    expect(isGrad(loss)).toBe(true);
    (loss as GradTensor).backward();
    expectClose(values(negative.grad as Tensor), flatNums(T.default.gn), 1e-9);
  });

  it("propagates NaN", () => {
    const loss = tripletMarginLoss(
      tensor([[Number.NaN, 0]], f64),
      tensor([[0, 0]], f64),
      tensor([[1, 1]], f64)
    );
    expect(Number.isNaN(values(loss)[0] as number)).toBe(true);
  });

  it("validates its arguments", () => {
    const [a, p, n] = plainTriple();
    expect(() => tripletMarginLoss(a, p, n, { reduction: "avg" as "mean" })).toThrow(
      InvalidParameterError
    );
    expect(() => tripletMarginLoss(a, p, n, { p: 0 })).toThrow(InvalidParameterError);
    expect(() => tripletMarginLoss(a, p, n, { p: -1 })).toThrow(InvalidParameterError);
    expect(() => tripletMarginLoss(a, p, n, { p: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => tripletMarginLoss(a, p, n, { eps: -1e-6 })).toThrow(InvalidParameterError);
    expect(() => tripletMarginLoss(a, p, n, { margin: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => tripletMarginLoss(a, p, tensor([[1, 2]], f64))).toThrow(ShapeError);
    expect(() => tripletMarginLoss(randn([2, 2, 2]), randn([2, 2, 2]), randn([2, 2, 2]))).toThrow(
      ShapeError
    );
    expect(() => tripletMarginLoss(tensor([["a"]]), tensor([["a"]]), tensor([["a"]]))).toThrow(
      /string dtype/
    );
    const [ga, gp, gn] = triple(true);
    expect(() => tripletMarginLoss(ga, gp, gn, { p: 0 })).toThrow(InvalidParameterError);
  });

  it("works on non-contiguous inputs", () => {
    const base = randn([4, 3], f64);
    const view = transpose(base);
    const copy = tensor(view.toArray() as number[][], f64);
    const other = randn([3, 4], f64);
    expectClose(
      values(tripletMarginLoss(view, other, copy, { margin: 5, reduction: "none" })),
      values(tripletMarginLoss(copy, other, copy, { margin: 5, reduction: "none" })),
      1e-12
    );
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("edge cases of the changed layers", () => {
  it("Linear handles an empty batch and rejects a 0-d input", () => {
    const layer = new Linear(3, 2);
    const out = layer.forward(tensor([], { dtype: "float32" }).reshape([0, 3]));
    expect(out.shape).toEqual([0, 2]);
    expect(() => layer.forward(tensor(1))).toThrow(ShapeError);
    expect(() => layer.forward(randn([2, 4]))).toThrow(ShapeError);
    expect(() => layer.forward(tensor([["a"]]))).toThrow(/string dtype/);
  });

  it("NaN input gives NaN output only in its own row", () => {
    const layer = new Linear(2, 2, f64);
    load(layer, { weight: [1, 0, 0, 1], bias: [0, 0] });
    const out = values(
      layer.forward(
        tensor(
          [
            [Number.NaN, 1],
            [2, 3],
          ],
          f64
        )
      )
    );
    expect(Number.isNaN(out[0] as number)).toBe(true);
    expect(out.slice(2)).toEqual([2, 3]);
  });

  it("Linear on a 1-D input and on a transposed view", () => {
    const layer = new Linear(3, 2, f64);
    load(layer, { weight: [1, 0, 0, 0, 1, 0], bias: [0, 0] });
    expectClose(values(layer.forward(tensor([5, 6, 7], f64))), [5, 6]);
    const base = tensor(
      [
        [1, 4],
        [2, 5],
        [3, 6],
      ],
      f64
    );
    expectClose(values(layer.forward(transpose(base))), [1, 2, 4, 5]);
  });

  it("Conv2d, LayerNorm and RNN accept non-contiguous inputs", () => {
    const conv = new Conv2d(1, 2, 2, f64);
    const base = randn([1, 4, 4, 1], f64);
    const view = transpose(base, [0, 3, 1, 2]);
    const copy = tensor(view.toArray() as number[][][][], f64);
    expectClose(values(conv.forward(view)), values(conv.forward(copy)), 1e-12);
    const ln = new LayerNorm(3, f64);
    const lnBase = randn([3, 2], f64);
    expectClose(
      values(ln.forward(transpose(lnBase))),
      values(ln.forward(tensor(transpose(lnBase).toArray() as number[][], f64))),
      1e-12
    );
    const rnn = new RNN(2, 3, f64);
    const seqBase = randn([2, 5], f64);
    expectClose(
      values(rnn.forward(transpose(seqBase))),
      values(rnn.forward(tensor(transpose(seqBase).toArray() as number[][], f64))),
      1e-12
    );
  });

  it("Dropout in evaluation mode returns the input values unchanged and plain", () => {
    const d = new Dropout(0.5);
    d.eval();
    const x = randn([3, 3]);
    const out = d.forward(x);
    expect(isGrad(out)).toBe(false);
    expect(values(out)).toEqual(values(x));
  });

  it("Flatten and Unflatten keep plain inputs plain and GradTensor inputs tracked", () => {
    const x = randn([2, 3, 2]);
    expect(isGrad(new Flatten().forward(x))).toBe(false);
    const g = GradTensor.fromTensor(x, { requiresGrad: true });
    const out = new Flatten().forward(g);
    out.sum().backward();
    expect(g.grad?.shape).toEqual([2, 3, 2]);
  });

  it("the training mode of BatchNorm still decides the statistics used", () => {
    const bn = new BatchNorm1d(2);
    const x = randn([8, 2]);
    bn.forward(x);
    bn.eval();
    const out = bn.forward(x);
    expect(out.shape).toEqual([8, 2]);
    expect(isGrad(out)).toBe(true);
  });
});

describe("review additions", () => {
  it("activations built from a scalar map give float32 for integer input, like the tensor operations", () => {
    const ints = tensor([[1, -2, 3]], { dtype: "int32" });
    for (const layer of [
      new Softplus(2),
      new SELU(),
      new Hardswish(),
      new Softsign(),
      new Sigmoid(),
    ]) {
      expect(layer.forward(ints).dtype).toBe("float32");
    }
    const f64 = tensor([[1, -2, 3]], { dtype: "float64" });
    expect(new Softplus(2).forward(f64).dtype).toBe("float64");
  });

  it("tripletMarginLoss with eps = 0 gives zero gradients, not NaN, for coinciding rows", () => {
    for (const p of [2, 1.5, 1]) {
      const a = GradTensor.fromTensor(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        { requiresGrad: true }
      );
      const positive = tensor([
        [1, 2],
        [3, 5],
      ]);
      const negative = tensor([
        [0, 0],
        [5, 5],
      ]);
      const loss = tripletMarginLoss(a, positive, negative, { eps: 0, p });
      loss.backward();
      expect(Array.from(a.grad?.data as Float32Array)).toEqual([0, 0, 0, 0]);
    }
  });
});
