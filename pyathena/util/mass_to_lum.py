from __future__ import print_function

import numpy as np
from scipy.interpolate import interp1d

from .piecewisepowerlaw import PiecewisePowerlaw

class mass_to_lum(object):
    """
    Simple class for calculating mass-to-light conversion factor

    """

    def __init__(self, model='Padova'):
        """
        Initialize a mass_to_lum object.

        Parameters
        ----------
        model: string
            Name of stellar evolutionary track.
            ``Padova`` - Power-law approximation to stellar luminosity, MS lifetime
            based on Padova evolutionary track (Bruzual & Charlot 2003).
            Data taken from Table 1 in Parravano et al. (2003).
        """

        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.LtoM_FUV_SB99, self.LtoM_PE_SB99, \
            self.LtoM_LW_SB99, self.QtoM_EUV_SB99 = self._get_MtoL_SB99()
        self.SNrate = self._get_SNrate_SB99()

        if model == 'Padova':
            self.calc_tMS, self.calc_LFUV, \
                self.calc_LH2, self.calc_Qi = self._get_MtoL_Padova()
            self.calc_ZAMS_mass = self._get_age_to_mass()

    def _get_age_to_mass(self):
        mass_range = np.array([1.2, 3.0, 6.0, 9.0, 12.0, 120.0])
        age_range = np.flip(self.calc_tMS(mass_range))

        def age_to_mass(age):
            # Isn't there more elegant way of doing this??
            age = np.atleast_1d(age)
            res = []
            for age_ in age:
                idx = np.where(age_ < age_range)[0][0]
                if idx == 0:
                    raise ValueError('Too small main sequence age')
                if idx == 1:
                    res.append(((age_ - 2.3 - 0.5)/7.6e2)**(-1.0/1.57))
                elif idx == 2:
                    res.append((age_/1.59e3)**(-1.0/1.81))
                elif idx == 3:
                    res.append((age_/2.76e3)**(-1.0/2.06))
                elif idx == 4:
                    res.append((age_/4.73e3)**(-1.0/2.36))
                elif idx == 5:
                    res.append((age_/4.73e3)**(-1.0/2.36))
                elif idx > 6:
                    res.append((age_/7.65e3)**(-1.0/2.8))

            return np.array(res)

        return age_to_mass

    def _get_MtoL_Padova(self):
        """
        Initialize power-law functions for model 'Padova'

        tMS: main sequence lifetime [Myr],
        LFUV: mean luminosities in the FUV band (912 - 2070A) [Lsun],
        LH2: mean luminosities in the H2 band (912 - 1100A) [Lsun]
        Qi: ionizing photon luminosity in the EUV band (< 912A).

        NOTE: Return NaN if mass is outside the valid range

        See Table 1 in Parravano et al. (2003).
        """

        # main sequence lifetime
        mass_range = np.array([1.2, 3.0, 6.0, 9.0, 12.0, 120.01])
        powers = np.array([-2.8, -2.36, -2.06, -1.81, -1.57])
        coeff = np.array([7.65e3, 4.73e3, 2.76e3, 1.59e3, 7.60e2])
        pp = PiecewisePowerlaw(mass_range, powers, coeff,
                               norm=False, externalval=None)

        def decorator(fn):
            def wrapper(x):
                x = np.atleast_1d(x)
                y = fn(x)
                y[x >= 12.0] += 2.3 + 0.5
                return y
            return wrapper
        tMS = decorator(pp)

        # FUV luminosity
        mass_range = np.array([1.8, 2.0, 2.5, 3.0, 6.0, 9.0, 12.0, 30.0, 120.01])
        powers = np.array([11.8, 9.03, 7.03, 4.76, 3.78, 3.31, 2.32, 1.54])
        coeff = np.array([2.77e-4, 1.88e-3, 1.19e-2, 1.47e-1, 8.22e-1, 2.29e0,
                          2.70e1, 3.99e2])
        LFUV = PiecewisePowerlaw(mass_range, powers, coeff,
                                 norm=False, externalval=None)

        # Lyman-Werner luminosity
        mass_range = np.array([1.8, 3.0, 4.0, 6.0, 9.0, 12.0, 15.0, 30.0, 120.01])
        powers = np.array([26.6, 13.7, 7.61, 5.13, 4.09, 3.43, 2.39, 1.69])
        coeff = np.array([1.94e-14, 2.86e-8, 1.35e-4, 1.10e-2, 1.09e-1, 5.47e-1, 9.09e0, 9.91e1])
        LH2 = PiecewisePowerlaw(mass_range, powers, coeff,
                                norm=False, externalval=None)

        # Ionizing photon rate
        mass_range = np.array([5.0, 7.0, 12.0, 20.0, 30.0, 40.0, 60.0, 120.01])
        powers = np.array([11.5, 8.87, 7.85, 4.91, 2.91, 2.23, 1.76])
        coeff = np.array([2.23e34, 3.69e36, 4.80e37, 3.12e41, 2.80e44, 3.49e45, 2.39e46])
        Qi = PiecewisePowerlaw(mass_range, powers, coeff,
                               norm=False, externalval=None)

        return tMS, LFUV, LH2, Qi

    def calc_LFUV_SB99(self, mass, age=0.0):
        return mass*self.LtoM_FUV_SB99(age)

    def calc_LPE_SB99(self, mass, age=0.0):
        return mass*self.LtoM_PE_SB99(age)

    def calc_LLW_SB99(self, mass, age=0.0):
        return mass*self.LtoM_LW_SB99(age)

    def calc_Qi_SB99(self, mass, age=0.0):
        return mass*self.QtoM_EUV_SB99(age)

    def _get_SNrate_SB99(self):

        _SNrate = [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                   0.00000000e+00,   0.00000000e+00,   4.75335226e-05,
                   5.52077439e-04,   5.22396189e-04,   5.48276965e-04,
                   5.33334895e-04,   5.44502653e-04,   5.79428696e-04,
                   5.58470195e-04,   5.94292159e-04,   6.10942025e-04,
                   5.98411595e-04,   6.39734835e-04,   6.50129690e-04,
                   6.28058359e-04,   5.79428696e-04,   5.22396189e-04,
                   5.06990708e-04,   4.72063041e-04,   4.77529274e-04,
                   4.83058802e-04,   4.75335226e-04,   4.50816705e-04,
                   4.42588372e-04,   4.34510224e-04,   4.27562886e-04,
                   4.19758984e-04,   4.13047502e-04,   4.07380278e-04,
                   4.00866718e-04,   3.99024902e-04,   3.99024902e-04,
                   3.94457302e-04,   3.89941987e-04,   3.85478358e-04,
                   3.81944271e-04,   3.77572191e-04,   3.74110588e-04,
                   3.69828180e-04,   3.66437575e-04,   3.63078055e-04,
                   3.59749335e-04,   3.55631319e-04,   3.52370871e-04,
                   3.49945167e-04,   3.46736850e-04,   3.43557948e-04,
                   3.40408190e-04,   3.37287309e-04,   3.34195040e-04,
                   3.31894458e-04,   3.26587832e-04,   3.22849412e-04,
                   3.19889511e-04,   3.16956746e-04,   3.14774831e-04,
                   3.11888958e-04,   3.09741930e-04,   3.06902199e-04,
                   3.04789499e-04,   3.01995172e-04,   2.99916252e-04,
                   2.97851643e-04,   2.95801247e-04,   2.93089325e-04,
                   2.91071712e-04,   2.89067988e-04,   2.87078058e-04,
                   2.85101827e-04,   2.83139200e-04,   2.81190083e-04,
                   2.79254384e-04,   2.77332010e-04,   2.75422870e-04,
                   2.73526873e-04,   2.71643927e-04,   2.69773943e-04,
                   2.68534445e-04,   2.66685866e-04,   2.66072506e-04,
                   2.64240876e-04,   2.63026799e-04,   2.61216135e-04,
                   2.60015956e-04,   2.58226019e-04,   2.57039578e-04,
                   2.55270130e-04,   2.54097271e-04,   2.52348077e-04,
                   2.51188643e-04,   2.50034536e-04,   2.48885732e-04,
                   2.47172415e-04,   2.46036760e-04,   2.44906324e-04,
                   2.43220401e-04,   2.42102905e-04,   2.40990543e-04,
                   2.39883292e-04,   2.38781128e-04,   2.37137371e-04,
                   2.36047823e-04,   2.34963282e-04,   2.33883724e-04,
                   2.32809126e-04,   2.31739465e-04,   2.30674719e-04,
                   2.29614865e-04,   2.28034207e-04,   2.27509743e-04,
                   2.26464431e-04,   2.25423921e-04,   2.24388192e-04,
                   2.23357222e-04,   2.22330989e-04,   2.21309471e-04,
                   2.20292646e-04,   2.19280494e-04,   2.18272991e-04,
                   2.17270118e-04,   2.16271852e-04,   2.15774441e-04,
                   2.14783047e-04,   2.13796209e-04,   2.12813905e-04,
                   2.12324446e-04,   2.10862815e-04,   2.10377844e-04,
                   2.09411246e-04,   2.08449088e-04,   2.07491352e-04,
                   2.07014135e-04,   2.06062991e-04,   2.05116218e-04,
                   2.04644464e-04,   2.03704208e-04,   2.02768272e-04,
                   2.02301918e-04,   2.01372425e-04,   2.00447203e-04,
                   1.99986187e-04,   1.99067334e-04,   1.98609492e-04,
                   1.97696964e-04,   1.90107828e-04,   1.89670592e-04,
                   1.88799135e-04,   1.87931682e-04,   1.87068214e-04,
                   1.86208714e-04,   1.85353162e-04,   1.84926862e-04,
                   1.83653834e-04,   1.83231442e-04,   1.82389570e-04,
                   1.81551566e-04,   1.80717413e-04,   1.80301774e-04,
                   1.79473363e-04,   1.78648757e-04,   1.78237877e-04,
                   1.77418948e-04,   1.76603782e-04,   1.75792361e-04,
                   1.75388050e-04,   1.74582215e-04,   1.73780083e-04,
                   1.72981636e-04,   1.72583789e-04,   1.71790839e-04,
                   1.71395731e-04,   1.70608239e-04,   1.69824365e-04,
                   1.69044093e-04,   1.68655303e-04,   1.68267406e-04,
                   1.67494288e-04,   1.66724721e-04,   1.65958691e-04,
                   1.65576996e-04,   1.64816239e-04,   1.64437172e-04,
                   1.63681652e-04,   1.63305195e-04,   1.62554876e-04]

        # cluster age in units of Myr
        dage = 0.2
        agemax = 40.0
        age = np.arange(0, agemax + dage, dage)
        SNrate = interp1d(age, _SNrate)
        return SNrate

    def _get_MtoL_SB99(self):

        _Psi_PE = [
              276.73506145, 283.29918113, 289.59841897, 296.81613976,
              304.72172851, 314.76067749, 325.39077734, 332.31456   ,
              350.89322934, 358.96975782, 373.90000171, 394.1703196 ,
              418.10218443, 447.88021923, 478.41878347, 466.8757721 ,
              428.15149686, 380.02548481, 341.31573599, 313.33984438,
              294.30488476, 271.42807864, 250.95320928, 234.17495415,
              220.56342607, 205.51554276, 192.82627906, 181.41766093,
              170.13294147, 159.71945965, 148.40504509, 136.73333272,
              126.21282738, 117.11777953, 109.51984036, 103.55136969,
              98.64696406,  94.21518038,  89.97252233,  86.09704998,
              82.42607853,  78.97819511,  75.76552534,  72.76479556,
              70.04444672,  67.50179017,  65.1410144 ,  62.89152219,
              60.81586399,  58.93331725,  57.06657363,  55.37471985,
              53.7362191 ,  52.19802312,  50.7310975 ,  49.34525754,
              48.08513666,  46.66242478,  45.24320324,  43.9241907 ,
              42.71104801,  41.582182  ,  40.5372192 ,  39.56760157,
              38.68399386,  37.8008295 ,  36.96862538,  36.17205055,
              35.39316702,  34.65074335,  33.94724503,  33.24581546,
              32.60009414,  31.91558317,  31.36610981,  30.74456452,
              30.18260048,  29.62105372,  29.07236919,  28.57992182,
              28.10525853,  27.61448218,  27.14981766,  26.73476077,
              26.30454709,  25.90153267,  25.47335631,  25.11989816,
              24.76694726,  24.39166714,  24.04236444,  23.71024825,
              23.37221258,  23.05669548,  22.73982757,  22.45902109,
              22.15169098,  21.86498403,  21.57076287,  21.30828651,
              21.04781474,  20.78505324,  20.52347016,  20.28153868,
              20.07622838,  19.83121021,  19.59169214,  19.36489572,
              19.16609119,  18.94384541,  18.73951035,  18.53594338,
              18.32702234,  18.12986416,  17.95855199,  17.75150223,
              17.57820304,  17.38950595,  17.21212441,  17.05222586,
              16.87585156,  16.74508104,  16.58312371,  16.41375466,
              16.27676558,  16.10917094,  15.9870481 ,  15.83539202,
              15.70739963,  15.5810877 ,  15.44406994,  15.32694164,
              15.1988409 ,  15.07732721,  14.94722914,  14.82143444,
              14.67067276,  14.5539274 ,  14.42577675,  14.28637533,
              14.15339744,  14.0631791 ,  13.96161938,  13.83269382,
              13.70516682,  13.59173448,  13.48055835,  13.36365434,
              13.26617453,  13.15442704,  13.04514251,  12.90503401,
              12.80516929,  12.67738108,  12.54559268,  12.43510645,
              12.32291023,  12.19856856,  12.08343559,  11.98547071,
              11.86952924,  11.7596549 ,  11.64722433,  11.55641497,
              11.4359627 ,  11.32846873,  11.25277497,  11.1682934 ,
              11.06575968,  10.95448475,  10.86160401,  10.77023236,
              10.67557977,  10.57930196,  10.49206925,  10.40819112,
              10.30860595,  10.22780082,  10.13490948,  10.05121174,
              9.95837592,   9.88629534,   9.79946059,   9.71492639,
              9.63435858,   9.54955613,   9.47859104,   9.39173226,
              9.32527106,   9.24730426,   9.1602775 ,   9.0892357 ,
              9.01670052,   8.94233507,   8.86974616,   8.80440874,
              8.7156227 ,   8.66156876,   8.60164256,   8.55078745,
              8.48449662
            ]

        _Psi_LW = [
            126.31816006, 129.72043083, 132.94334281, 136.51605735,
            140.79567861, 146.31862466, 153.53038347, 158.15372208,
            163.45690499, 169.71666429, 178.82366525, 194.47774438,
            203.42768527, 205.30451464, 207.33168602, 200.87617278,
            186.29497956, 166.34465084, 152.17177334, 142.91855617,
            132.5245931 , 122.85839695, 114.27536916, 106.5422258 ,
            98.9371041 ,  90.69353298,  84.01331463,  77.90676252,
            71.85588805,  64.97833395,  58.03744736,  52.49361222,
            48.06273321,  44.96785832,  42.28911578,  39.76936977,
            37.45539713,  35.33351678,  33.35854207,  31.53407898,
            29.92236456,  28.48047947,  27.15718794,  25.90878799,
            24.79451991,  23.75002916,  22.77698972,  21.83785626,
            20.98119391,  20.21869671,  19.44645928,  18.74934486,
            18.07574181,  17.44924312,  16.8313626 ,  16.27484329,
            15.7600669 ,  15.22190135,  14.71950805,  14.25050947,
            13.81470259,  13.38135529,  12.97281738,  12.58687855,
            12.23204733,  11.87029753,  11.53514083,  11.21334971,
            10.89887047,  10.59486113,  10.31257689,  10.03290317,
            9.77667082,   9.4982971 ,   9.28871048,   9.04296723,
            8.83004699,   8.61021522,   8.39610563,   8.20477914,
            8.02455839,   7.83225318,   7.65167439,   7.49559425,
            7.328053  ,   7.1735738 ,   7.0111111 ,   6.87873251,
            6.7495921 ,   6.60551874,   6.47546527,   6.3537924 ,
            6.22762012,   6.11295166,   5.99413212,   5.89391041,
            5.77623137,   5.66742972,   5.56066295,   5.46563598,
            5.37208183,   5.2770809 ,   5.18202402,   5.09505382,
            5.02516536,   4.93557335,   4.85061097,   4.76739094,
            4.69874869,   4.61816634,   4.54891437,   4.47485309,
            4.39954733,   4.32915362,   4.26749513,   4.19467717,
            4.13475314,   4.06589554,   4.00304779,   3.94754387,
            3.88314134,   3.83950119,   3.7814571 ,   3.72218038,
            3.67452403,   3.61235961,   3.56899879,   3.51502836,
            3.46696721,   3.42125989,   3.36987174,   3.32519474,
            3.27658007,   3.22880401,   3.18138855,   3.13569554,
            3.08100481,   3.0389194 ,   2.99242885,   2.94221858,
            2.89596357,   2.8666213 ,   2.82873773,   2.78590513,
            2.74161791,   2.70065343,   2.65862175,   2.62478264,
            2.58535051,   2.5465942 ,   2.51721097,   2.47518108,
            2.44005466,   2.40658471,   2.36998407,   2.33618169,
            2.30132627,   2.2682113 ,   2.23253863,   2.20815774,
            2.17609782,   2.14180464,   2.11239343,   2.08847961,
            2.05462582,   2.02140553,   2.00497477,   1.98350754,
            1.95558307,   1.92420788,   1.8990255 ,   1.87571721,
            1.84874276,   1.82411981,   1.80116501,   1.7792482 ,
            1.75210308,   1.73161692,   1.70625195,   1.68362244,
            1.65760979,   1.64166528,   1.61802691,   1.59659553,
            1.57213814,   1.55370273,   1.53595847,   1.51139798,
            1.49546522,   1.4753886 ,   1.45381468,   1.4360342 ,
            1.41736498,   1.3984063 ,   1.37986945,   1.36392047,
            1.34221563,   1.33009491,   1.3156304 ,   1.30483779,
            1.28902265
        ]


        # SB99 specific FUV luminosity Psi = L/M_* as a function of age
        _Psi = [453.08679514, 464.53452871, 475.47798641, 485.7073231 ,
                496.9356504 , 511.96949628, 548.4466934 , 562.63899134,
                592.79623836, 615.30727212, 653.51217536, 720.24242122,
                763.78785281, 786.69155062, 801.01170583, 761.35985867,
                667.88663137, 593.6170566 , 538.91202362, 497.36625759,
                457.87623408, 419.12111464, 388.81524695, 365.90999537,
                346.11651385, 316.07389538, 292.94186317, 274.96469223,
                258.15397177, 239.26669601, 220.01253422, 199.85879126,
                182.25156179, 169.97567772, 159.78310952, 151.33661229,
                143.66317928, 136.40155234, 129.42792748, 122.70219911,
                117.19632053, 112.83371757, 108.73964512, 104.81626952,
                101.08554852,  97.48501856,  94.0111926 ,  90.65558598,
                87.45711133,  84.36600081,  81.42197733,  78.57572024,
                75.82079324,  73.15775994,  70.59190409,  68.11345756,
                65.74853714,  63.83016055,  62.05007971,  60.33343415,
                58.67123283,  57.08305372,  55.58135509,  54.1725337 ,
                52.83354666,  51.55583727,  50.32801935,  49.13451734,
                47.97308984,  46.84956744,  45.76005756,  44.69961541,
                43.67094132,  42.67124003,  41.69927098,  40.75995245,
                39.85595526,  38.96624566,  38.14938374,  37.52161742,
                36.90927209,  36.30833196,  35.72438727,  35.15214723,
                34.59375653,  34.05152912,  33.52439301,  33.01132771,
                32.51351216,  32.03548703,  31.56658689,  31.10721136,
                30.65638137,  30.21401644,  29.7786161 ,  29.35306829,
                28.93444047,  28.52413306,  28.12153029,  27.72770005,
                27.34063726,  26.96049212,  26.59058388,  26.2286874 ,
                25.87839393,  25.53930949,  25.20577962,  24.87753233,
                24.55532672,  24.2366395 ,  23.92356405,  23.61665259,
                23.31338004,  23.01438386,  22.72099394,  22.4325984 ,
                22.14957227,  21.87161268,  21.60064799,  21.3345638 ,
                21.07459035,  20.82300964,  20.58119904,  20.3466166 ,
                20.11706604,  19.89354325,  19.67405505,  19.46068272,
                19.25034161,  19.04528147,  18.84141499,  18.63901451,
                18.43885467,  18.24088716,  18.04310648,  17.84790554,
                17.65578097,  17.46609886,  17.28120726,  17.09724708,
                16.91690797,  16.737069  ,  16.56133404,  16.38751444,
                16.21443932,  16.04098275,  15.867691  ,  15.6937046 ,
                15.51995743,  15.34695103,  15.17566247,  15.01156431,
                14.85040033,  14.69095164,  14.53397608,  14.37892192,
                14.22670176,  14.07635124,  13.92875437,  13.78274391,
                13.63917075,  13.4973675 ,  13.35813056,  13.22123433,
                13.08649759,  12.95453139,  12.8246124 ,  12.69954539,
                12.57635968,  12.45484991,  12.33461741,  12.21537161,
                12.09724828,  11.98061338,  11.86500289,  11.75089772,
                11.63743786,  11.52500048,  11.41363433,  11.30378932,
                11.19527526,  11.08828425,  10.9831809 ,  10.87910342,
                10.77653312,  10.67527811,  10.57519197,  10.47638608,
                10.37855252,  10.28201134,  10.18685478,  10.09308863,
                10.00041292,   9.90877628,   9.81816843,   9.72881811,
                9.64106077,   9.55428174,   9.46869859,   9.38459354,
                9.30185274]

        # SB99 specific EUV luminosity Xi = Qi/M_* as a function of age
        # _Xi= [4.12097519e+46,   4.09260660e+46,   4.09260660e+46,
        #      4.10204103e+46,   4.11149721e+46,   4.12097519e+46,
        #       4.11149721e+46,   4.11149721e+46,   4.11149721e+46,
        #       4.10204103e+46,   4.08319386e+46,   3.99944750e+46,
        #       4.04575892e+46,   4.08319386e+46,   4.11149721e+46,
        #       4.07380278e+46,   3.93550075e+46,   3.99944750e+46,
        #       3.98107171e+46,   3.97191549e+46,   3.85478358e+46,
        #       3.72391706e+46,   3.51560441e+46,   3.35737614e+46,
        #       3.14774831e+46,   2.83791903e+46,   2.61818301e+46,
        #       2.33883724e+46,   2.05116218e+46,   1.96336028e+46,
        #       2.28559880e+46,   2.48313311e+46,   2.54097271e+46,
        #       2.36591970e+46,   2.18272991e+46,   2.02768272e+46,
        #       1.88799135e+46,   1.71790839e+46,   1.56314764e+46,
        #       1.29121927e+46,   1.18304156e+46,   1.08893009e+46,
        #       9.97700064e+45,   9.37562007e+45,   8.47227414e+45,
        #       7.87045790e+45,   7.22769804e+45,   5.86138165e+45,
        #       5.59757601e+45,   5.15228645e+45,   4.68813382e+45,
        #       4.15910610e+45,   3.53183170e+45,   3.22849412e+45,
        #       2.84446111e+45,   2.55270130e+45,   2.31739465e+45,
        #       2.14289060e+45,   1.87931682e+45,   1.70608239e+45,
        #       1.53461698e+45,   1.38675583e+45,   1.26765187e+45,
        #       1.15080039e+45,   1.04231743e+45,   9.44060876e+44,
        #       8.57037845e+44,   7.87045790e+44,   7.24435960e+44,
        #       6.65273156e+44,   6.13762005e+44,   5.63637656e+44,
        #       5.19995997e+44,   4.77529274e+44,   4.33510878e+44,
        #       4.04575892e+44,   3.74973002e+44,   3.49140315e+44,
        #       3.22849412e+44,   2.99226464e+44,   2.79898132e+44,
        #       2.61818301e+44,   2.45470892e+44,   2.28559880e+44,
        #       2.13304491e+44,   1.99067334e+44,   1.88799135e+44,
        #       1.78648757e+44,   1.68655303e+44,   1.58854675e+44,
        #       1.49968484e+44,   1.41579378e+44,   1.34276496e+44,
        #       1.27938130e+44,   1.21059813e+44,   1.14287833e+44,
        #       1.08893009e+44,   1.02801630e+44,   9.79489985e+43,
        #       9.31107875e+43,   8.89201118e+43,   8.49180475e+43,
        #       8.12830516e+43,   7.70903469e+43,   7.44731974e+43,
        #       7.04693069e+43,   6.71428853e+43,   6.41209577e+43,
        #       6.09536897e+43,   5.88843655e+43,   5.57185749e+43,
        #       5.32108259e+43,   5.17606832e+43,   4.96592321e+43,
        #       4.73151259e+43,   4.56036916e+43,   4.38530698e+43,
        #       4.20726628e+43,   4.00866718e+43,   3.88150366e+43,
        #       3.71535229e+43,   3.57272838e+43,   3.45939378e+43,
        #       3.30369541e+43,   3.17687407e+43,   3.02691343e+43,
        #       2.96483139e+43,   2.83791903e+43,   2.74157417e+43,
        #       2.67300641e+43,   2.54097271e+43,   2.46036760e+43,
        #       2.37137371e+43,   2.30144182e+43,   2.21819642e+43,
        #       2.16271852e+43,   2.07491352e+43,   2.00909281e+43,
        #       1.91866874e+43,   1.87931682e+43,   1.79887092e+43,
        #       1.75388050e+43,   1.68267406e+43,   1.64437172e+43,
        #       1.58489319e+43,   1.52054753e+43,   1.46554784e+43,
        #       1.43218790e+43,   1.40604752e+43,   1.36144468e+43,
        #       1.32434154e+43,   1.28528666e+43,   1.24738351e+43,
        #       1.21618600e+43,   1.17760597e+43,   1.14551294e+43,
        #       1.11429453e+43,   1.08143395e+43,   1.06169556e+43,
        #       1.02565193e+43,   1.00230524e+43,   9.74989638e+42,
        #       9.41889597e+42,   9.09913273e+42,   8.91250938e+42,
        #       8.60993752e+42,   8.45278845e+42,   8.24138115e+42,
        #       7.99834255e+42,   7.78036551e+42,   7.63835784e+42,
        #       7.39605275e+42,   7.17794291e+42,   7.07945784e+42,
        #       6.90239804e+42,   6.79203633e+42,   6.59173895e+42,
        #       6.51628394e+42,   6.29506183e+42,   6.19441075e+42,
        #       6.09536897e+42,   5.95662144e+42,   5.82103218e+42,
        #       5.66239289e+42,   5.58470195e+42,   5.45757861e+42,
        #       5.33334895e+42,   5.19995997e+42,   5.12861384e+42,
        #       5.03500609e+42,   4.93173804e+42,   4.74241985e+42,
        #       4.68813382e+42,   4.62381021e+42,   4.44631267e+42,
        #       4.41570447e+42,   4.29536427e+42,   4.20726628e+42,
        #       4.11149721e+42,   4.01790811e+42,   3.97191549e+42,
        #       3.87257645e+42,   3.80189396e+42,   3.72391706e+42,
        #       3.67282300e+42,   3.57272838e+42,   3.51560441e+42,
        #       3.43557948e+42,   3.41192912e+42,   3.35737614e+42,
        #       3.31131121e+42,   3.23593657e+42,   3.16956746e+42,
        #       3.13328572e+42,   3.07609681e+42,   3.01995172e+42,
        #       2.97166603e+42,   2.89734359e+42,   2.87078058e+42,
        #       2.83139200e+42,   2.79898132e+42,   2.71019163e+42,
        #       2.67300641e+42,   2.62421854e+42,   2.58226019e+42,
        #       2.52929800e+42,   2.47742206e+42,   2.43220401e+42,
        #       2.40990543e+42,   2.35504928e+42,   2.31739465e+42,
        #       2.28034207e+42,   2.23357222e+42,   2.20800473e+42,
        #       2.15774441e+42,   2.11836114e+42,   2.07491352e+42,
        #       2.04173794e+42,   2.00447203e+42,   1.97242274e+42,
        #       1.94088588e+42,   1.90546072e+42,   1.89670592e+42,
        #       1.87068214e+42,   1.84077200e+42,   1.80717413e+42,
        #       1.78237877e+42,   1.74582215e+42,   1.72583789e+42,
        #       1.70215851e+42,   1.67109061e+42,   1.65958691e+42,
        #       1.62181010e+42,   1.58489319e+42,   1.58124804e+42,
        #       1.55596563e+42,   1.52405275e+42,   1.49623566e+42,
        #       1.48251809e+42,   1.44877185e+42,   1.43548943e+42,
        #       1.40928880e+42,   1.39315680e+42,   1.36144468e+42,
        #       1.33659552e+42,   1.31825674e+42,   1.30316678e+42,
        #       1.26765187e+42,   1.25602996e+42,   1.23594743e+42,
        #       1.21338885e+42,   1.18576875e+42,   1.17489755e+42,
        #       1.14815362e+42,   1.14024979e+42,   1.11429453e+42,
        #       1.09395637e+42,   1.07151931e+42,   1.05925373e+42,
        #       1.03992017e+42,   1.01859139e+42,   1.00925289e+42,
        #       1.00000000e+42,   9.81747943e+41,   9.68277856e+41,
        #       9.61612278e+41,   9.50604794e+41,   9.31107875e+41,
        #       9.14113241e+41,   8.95364766e+41,   8.83079900e+41,
        #       8.70963590e+41,   8.57037845e+41,   8.47227414e+41,
        #       8.43334758e+41,   8.22242650e+41,   8.09095899e+41,
        #       7.99834255e+41,   7.87045790e+41,   7.72680585e+41,
        #       7.74461798e+41,   7.53355564e+41,   7.36207097e+41,
        #       7.26105957e+41,   7.09577768e+41,   7.09577768e+41,
        #       7.01455298e+41,   6.96626514e+41,   6.71428853e+41,
        #       6.60693448e+41,   6.51628394e+41,   6.41209577e+41,
        #       6.23734835e+41,   6.23734835e+41,   6.12350392e+41,
        #       5.99791076e+41,   5.86138165e+41,   5.83445104e+41,
        #       5.75439937e+41,   5.64936975e+41,   5.61047976e+41,
        #       5.50807696e+41,   5.35796658e+41,   5.32108259e+41,
        #       5.22396189e+41,   5.12861384e+41,   5.10505000e+41,
        #       5.05824662e+41,   4.93173804e+41,   4.84172368e+41,
        #       4.70977326e+41,   4.72063041e+41,   4.67735141e+41,
        #       4.61317575e+41,   4.57088190e+41,   4.52897580e+41,
        #       4.47713304e+41,   4.39541615e+41,   4.31519077e+41,
        #       4.27562886e+41,   4.19758984e+41,   4.13999675e+41,
        #       4.11149721e+41,   4.03645393e+41,   3.95366620e+41,
        #       3.93550075e+41,   3.89045145e+41,   3.88150366e+41,
        #       3.78442585e+41,   3.70680722e+41,   3.68977599e+41,
        #       3.65594792e+41,   3.58096437e+41,   3.53183170e+41,
        #       3.53183170e+41,   3.41192912e+41,   3.37287309e+41,
        #       3.36511569e+41,   3.27340695e+41,   3.25087297e+41,
        #       3.16227766e+41,   3.14774831e+41,   3.14050869e+41,
        #       3.11171634e+41,   3.00607630e+41,   2.96483139e+41,
        #       2.93089325e+41,   2.87078058e+41,   2.78612117e+41,
        #       2.79898132e+41,   2.77332010e+41,   2.71019163e+41,
        #       2.70395836e+41,   2.66072506e+41,   2.57632116e+41,
        #       2.56448404e+41,   2.52929800e+41,   2.47742206e+41,
        #       2.44906324e+41,   2.39331576e+41,   2.38781128e+41,
        #       2.33883724e+41,   2.32273680e+41,   2.28034207e+41,
        #       2.25423921e+41,   2.18776162e+41,   2.17770977e+41,
        #       2.14783047e+41,   2.11348904e+41,   2.08449088e+41,
        #       2.05589060e+41,   2.04173794e+41,   1.99526231e+41,
        #       1.96336028e+41,   1.97696964e+41,   1.94088588e+41,
        #       1.91866874e+41,   1.89670592e+41,   1.90546072e+41,
        #       1.87931682e+41,   1.84926862e+41,   1.83231442e+41,
        #       1.83231442e+41,   1.77827941e+41,   1.73780083e+41,
        #       1.72981636e+41,   1.71001532e+41,   1.67880402e+41,
        #       1.66341265e+41,   1.64058977e+41,   1.61435856e+41,
        #       1.62554876e+41,   1.58854675e+41,   1.55596563e+41,
        #       1.58124804e+41,   1.53815464e+41,   1.51705037e+41,
        #       1.48593564e+41,   1.47231250e+41,   1.47910839e+41,
        #       1.47910839e+41,   1.41579378e+41,   1.39958732e+41,
        #       1.36144468e+41,   1.35518941e+41,   1.34896288e+41,
        #       1.32739446e+41,   1.30918192e+41,   1.29419584e+41,
        #       1.27938130e+41,   1.26182753e+41,   1.25892541e+41,
        #       1.23026877e+41,   1.21898960e+41,   1.22743923e+41,
        #       1.18304156e+41,   1.17489755e+41,   1.18576875e+41,
        #       1.16412603e+41,   1.16144861e+41,   1.11173173e+41,
        #       1.10153931e+41,   1.08893009e+41,   1.06659612e+41,
        #       1.06169556e+41,   1.04472022e+41,   1.04231743e+41,
        #       1.02329299e+41,   1.01157945e+41,   9.95405417e+40,
        #       9.63829024e+40,   9.54992586e+40,   9.44060876e+40,
        #       9.41889597e+40,   9.33254301e+40,   9.22571427e+40,
        #       9.12010839e+40,   8.99497582e+40,   8.72971368e+40,
        #       8.70963590e+40,   8.59013522e+40,   8.45278845e+40,
        #       8.41395142e+40,   8.29850768e+40,   8.22242650e+40,
        #       8.18464788e+40,   8.01678063e+40,   7.81627805e+40,
        #       7.76247117e+40,   7.69130440e+40,   7.49894209e+40,
        #       7.46448758e+40,   7.26105957e+40,   7.19448978e+40,
        #       7.16143410e+40,   6.99841996e+40,   6.96626514e+40,
        #       6.80769359e+40,   6.63743070e+40,   6.71428853e+40,
        #       6.68343918e+40,   6.62216504e+40,   6.79203633e+40,
        #       6.48634434e+40,   6.45654229e+40,   6.36795521e+40,
        #       6.33869711e+40,   6.29506183e+40,   6.25172693e+40,
        #       6.18016400e+40,   6.13762005e+40,   6.26613865e+40,
        #       6.01173737e+40,   5.94292159e+40,   5.88843655e+40,
        #       5.74116462e+40,   5.72796031e+40,   5.67544605e+40,
        #       5.76766463e+40,   5.70164272e+40]

        _Xi= [4.73450983e+46, 4.69166169e+46, 4.69735447e+46, 4.69663822e+46,
              4.68113425e+46, 4.61690473e+46, 4.52697441e+46, 4.56713558e+46,
              4.34349274e+46, 4.35170935e+46, 4.17274880e+46, 3.72768914e+46,
              3.29801369e+46, 2.73206323e+46, 2.10118212e+46, 2.40352938e+46,
              2.80657297e+46, 2.44282760e+46, 2.12936261e+46, 1.68145220e+46,
              1.21862516e+46, 9.91760497e+45, 8.28261083e+45, 6.97069389e+45,
              5.53148513e+45, 4.52982583e+45, 3.32786061e+45, 2.65728032e+45,
              2.16210762e+45, 1.76927172e+45, 1.44754822e+45, 1.19421899e+45,
              9.82586353e+44, 8.04466000e+44, 6.76511593e+44, 5.72325417e+44,
              4.83171327e+44, 4.01889399e+44, 3.45158160e+44, 2.96378130e+44,
              2.56532873e+44, 2.23845335e+44, 1.94370228e+44, 1.71206483e+44,
              1.52694570e+44, 1.35314031e+44, 1.21018031e+44, 1.08741945e+44,
              9.77643037e+43, 8.77158533e+43, 7.94607922e+43, 7.25071575e+43,
              6.63297214e+43, 5.97038296e+43, 5.40634130e+43, 4.93988414e+43,
              4.57984749e+43, 4.18251355e+43, 3.87269539e+43, 3.53332955e+43,
              3.27278726e+43, 3.04145740e+43, 2.79328072e+43, 2.59911022e+43,
              2.40218973e+43, 2.22267014e+43, 2.07407823e+43, 1.93900213e+43,
              1.81204802e+43, 1.67420628e+43, 1.56629948e+43, 1.46294638e+43,
              1.37598868e+43, 1.27217307e+43, 1.21957054e+43, 1.14820640e+43,
              1.08176314e+43, 1.01823446e+43, 9.62981291e+42, 9.17621396e+42,
              8.64952222e+42, 8.12786072e+42, 7.68691700e+42, 7.28862918e+42,
              6.89098886e+42, 6.56975468e+42, 6.16815846e+42, 5.92767161e+42,
              5.66325754e+42, 5.40322067e+42, 5.22267978e+42, 4.98667890e+42,
              4.77866126e+42, 4.56938751e+42, 4.38570061e+42, 4.22148288e+42,
              4.01286246e+42, 3.79893274e+42, 3.66339134e+42, 3.50555041e+42,
              3.39064171e+42, 3.23904223e+42, 3.13011460e+42, 2.99410418e+42,
              2.90610672e+42, 2.81906741e+42, 2.70040982e+42, 2.61627656e+42,
              2.52690342e+42, 2.44205249e+42, 2.37570245e+42, 2.26952208e+42,
              2.19378919e+42, 2.10413667e+42, 2.04429913e+42, 1.96322703e+42,
              1.89478119e+42, 1.82754725e+42, 1.75719812e+42, 1.69501939e+42,
              1.64045584e+42, 1.60246027e+42, 1.55732639e+42, 1.50813242e+42,
              1.45768107e+42, 1.41108833e+42, 1.37038916e+42, 1.33542436e+42,
              1.28772926e+42, 1.24992885e+42, 1.21084530e+42, 1.17605361e+42,
              1.12664745e+42, 1.09669488e+42, 1.05738484e+42, 1.02133288e+42,
              9.88329204e+41, 9.59581702e+41, 9.21021373e+41, 8.91583507e+41,
              8.57048778e+41, 8.39110552e+41, 8.14312740e+41, 7.98019526e+41,
              7.68378207e+41, 7.41327297e+41, 7.19453298e+41, 7.06783231e+41,
              6.78590074e+41, 6.60230980e+41, 6.49281706e+41, 6.18089680e+41,
              5.94956750e+41, 5.87113292e+41, 5.63120795e+41, 5.45270956e+41,
              5.22394641e+41, 5.12598479e+41, 4.90349960e+41, 4.81175336e+41,
              4.68876976e+41, 4.48039610e+41, 4.36447961e+41, 4.26548175e+41,
              4.12581781e+41, 3.93951132e+41, 3.90674651e+41, 3.82044682e+41,
              3.73481689e+41, 3.60427250e+41, 3.49983704e+41, 3.42851502e+41,
              3.30216183e+41, 3.24190917e+41, 3.15478840e+41, 3.07374889e+41,
              2.98709640e+41, 2.94201663e+41, 2.81125550e+41, 2.72554179e+41,
              2.63654346e+41, 2.61199457e+41, 2.50683584e+41, 2.43843716e+41,
              2.32266793e+41, 2.30782741e+41, 2.24869049e+41, 2.14514595e+41,
              2.10361588e+41, 2.03844461e+41, 1.98708328e+41, 1.92986201e+41,
              1.87211715e+41, 1.81224092e+41, 1.75482500e+41, 1.70754522e+41,
              1.65606167e+41, 1.63926973e+41, 1.59404697e+41, 1.58186530e+41,
              1.53622887e+41]


        # cluster age in units of Myr
        dage = 0.2
        agemax = 40.0
        age = np.arange(0, agemax + dage, dage)

        Psi = interp1d(age, _Psi)
        Psi_PE = interp1d(age, _Psi_PE)
        Psi_LW = interp1d(age, _Psi_LW)

        # dage = 0.1
        # agemax = 50.0
        dage = 0.2
        agemax = 40.0
        age = np.arange(0.0, agemax + dage, dage)
        Xi = interp1d(age, _Xi)


        return (Psi, Psi_PE, Psi_LW, Xi)