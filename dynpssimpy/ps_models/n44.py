def load():
    return {
        'base_mva': 1000,
        'f': 50,
        'slack_bus': '3300',

        'buses': [
            ['name', 'V_n', 'Area', 'V_0'],
            ['3000', 420, 23, 0.98639],
            ['3020', 420, 23, 0.9826],
            ['3100', 420, 22, 0.96568],
            ['3115', 420, 21, 1],
            ['3200', 420, 23, 0.97198],
            ['3244', 300, 22, 1.00143],
            ['3245', 420, 22, 1],
            ['3249', 420, 21, 1],
            ['3300', 420, 23, 1],
            ['3359', 420, 23, 0.97417],
            ['3360', 135, 23, 0.97064],
            ['3701', 300, 21, 1.01192],
            ['5100', 300, 16, 1],
            ['5101', 420, 16, 0.98345],
            ['5102', 420, 16, 0.98578],
            ['5103', 420, 16, 0.98504],
            ['5300', 300, 15, 1],
            ['5301', 420, 15, 0.9936],
            ['5304', 420, 15, 0.98801],
            ['5305', 420, 15, 0.99511],
            ['5400', 300, 12, 1.007],
            ['5401', 420, 12, 0.99321],
            ['5402', 420, 12, 1.0022],
            ['5500', 300, 11, 1.004],
            ['5501', 420, 11, 1.00245],
            ['5600', 300, 13, 1.01],
            ['5601', 300, 13, 1.01809],
            ['5602', 420, 13, 0.98532],
            ['5603', 300, 13, 0.95282],
            ['5610', 300, 13, 0.95049],
            ['5620', 300, 13, 1.00896],
            ['6000', 300, 14, 1.005],
            ['6001', 420, 14, 1.00022],
            ['6100', 300, 14, 1],
            ['6500', 300, 17, 1],
            ['6700', 300, 18, 1.02],
            ['6701', 420, 18, 1.00754],
            ['7000', 420, 32, 1],
            ['7010', 420, 32, 0.99636],
            ['7020', 420, 32, 1.00002],
            ['7100', 420, 31, 1],
            ['8500', 420, 24, 0.96301],
            ['8600', 420, 24, 0.96294],
            ['8700', 420, 24, 0.963],
        ],

        'lines': [
                ['name',        'from_bus', 'to_bus', 'length', 'S_n', 'V_n', 'unit', 'R', 'X', 'B'],
                ['L3000-3020', '3000',      '3020',     1,      0,      0,      'p.u.',     0, 0.006, 0],
                ['L3000-3115', '3000',      '3115',     1,      0,      0,      'p.u.',     0.045, 0.54, 0.5],
                ['L3000-3245-1', '3000',      '3245',     1,      0,      0,      'p.u.',     0.0048, 0.072, 0.05],
                ['L3000-3245-2', '3000',      '3245',     1,      0,      0,      'p.u.',     0.0108, 0.12, 0.05],
                ['L3000-3300-1', '3000',      '3300',     1,      0,      0,      'p.u.',     0.0036, 0.048, 0.03],
                ['L3000-3300-2', '3000',      '3300',     1,      0,      0,      'p.u.',     0.0054, 0.06, 0.025],
                ['L3100-3115', '3100',      '3115',     1,      0,      0,      'p.u.',     0.018, 0.24, 0.11],
                ['L3100-3200-1', '3100',      '3200',     1,      0,      0,      'p.u.',     0.024, 0.144, 0.2],
                ['L3100-3200-2', '3100',      '3200',     1,      0,      0,      'p.u.',     0.024, 0.144, 0.2],
                ['L3100-3200-3', '3100',      '3200',     1,      0,      0,      'p.u.',     0.024, 0.144, 0.2],
                ['L3100-3249', '3100',      '3249',     1,      0,      0,      'p.u.',     0.018, 0.258, 0.16],
                ['L3100-3359-1', '3100',      '3359',     1,      0,      0,      'p.u.',     0.048, 0.3, 0.25],
                ['L3100-3359-2', '3100',      '3359',     1,      0,      0,      'p.u.',     0.024, 0.138, 0.24],
                ['L3115-3245', '3115',      '3245',     1,      0,      0,      'p.u.',     0.027, 0.3, 0.14],
                ['L3115-3249', '3115',      '3249',     1,      0,      0,      'p.u.',     0.009, 0.12, 0.08],
                ['L3115-6701', '3115',      '6701',     1,      0,      0,      'p.u.',     0.024, 0.24, 0.1],
                ['L3115-7100', '3115',      '7100',     1,      0,      0,      'p.u.',     0.024, 0.078, 0.13],
                ['L3200-3300', '3200',      '3300',     1,      0,      0,      'p.u.',     0.012, 0.12, 0.06],
                ['L3200-3359', '3200',      '3359',     1,      0,      0,      'p.u.',     0.006, 0.12, 0.07],
                ['L3200-8500', '3200',      '8500',     1,      0,      0,      'p.u.',     0.006, 0.102, 0.06],
                ['L3244-6500', '3244',      '6500',     1,      0,      0,      'p.u.',     0.006, 0.12, 0.06],
                ['L3249-7100', '3249',      '7100',     1,      0,      0,      'p.u.',     0.012, 0.045, 0.078],
                ['L3300-8500-1', '3300',      '8500',     1,      0,      0,      'p.u.',     0.012, 0.138, 0.06],
                ['L3300-8500-2', '3300',      '8500',     1,      0,      0,      'p.u.',     0.0072, 0.162, 0.1],
                ['L3359-5101-1', '3359',      '5101',     1,      0,      0,      'p.u.',     0.0096, 0.156, 0.09],
                ['L3359-5101-2', '3359',      '5101',     1,      0,      0,      'p.u.',     0.012, 0.132, 0.06],
                ['L3359-8500-1', '3359',      '8500',     1,      0,      0,      'p.u.',     0.0072, 0.162, 0.1],
                ['L3359-8500-2', '3359',      '8500',     1,      0,      0,      'p.u.',     0.015, 0.192, 0.09],
                ['L3701-6700', '3701',      '6700',     1,      0,      0,      'p.u.',     0.15, 1.2, 0.03],
                ['L5100-5500', '5100',      '5500',     1,      0,      0,      'p.u.',     0.0162, 0.156, 0.044],
                ['L5100-6500', '5100',      '6500',     1,      0,      0,      'p.u.',     0.048, 0.54, 0.06],
                ['L5101-5102', '5101',      '5102',     1,      0,      0,      'p.u.',     0.0048, 0.06, 0.09],
                ['L5101-5103', '5101',      '5103',     1,      0,      0,      'p.u.',     0.006, 0.084, 0.04],
                ['L5101-5501', '5101',      '5501',     1,      0,      0,      'p.u.',     0.006, 0.09, 0.55],
                ['L5102-5103', '5102',      '5103',     1,      0,      0,      'p.u.',     0.0024, 0.042, 0.03],
                ['L5102-5304', '5102',      '5304',     1,      0,      0,      'p.u.',     0.0102, 0.144, 0.07],
                ['L5102-6001', '5102',      '6001',     1,      0,      0,      'p.u.',     0.018, 0.276, 0.13],
                ['L5103-5304-1', '5103',      '5304',     1,      0,      0,      'p.u.',     0.012, 0.15, 0.07],
                ['L5103-5304-2', '5103',      '5304',     1,      0,      0,      'p.u.',     0.0078, 0.12, 0.06],
                ['L5300-6100', '5300',      '6100',     1,      0,      0,      'p.u.',     0.0126, 0.132, 0.01],
                ['L5301-5304', '5301',      '5304',     1,      0,      0,      'p.u.',     0.006, 0.12, 0.06],
                ['L5301-5305', '5301',      '5305',     1,      0,      0,      'p.u.',     0.0042, 0.072, 0.031],
                ['L5301-6001', '5301',      '6001',     1,      0,      0,      'p.u.',     0.0078, 0.12, 0.05],
                ['L5304-5305-1', '5304',      '5305',     1,      0,      0,      'p.u.',     0.006, 0.09, 0.05],
                ['L5304-5305-2', '5304',      '5305',     1,      0,      0,      'p.u.',     0.0078, 0.0102, 0.04],
                ['L5400-5500', '5400',      '5500',     1,      0,      0,      'p.u.',     0.0054, 0.564, 0.05],
                ['L5400-6000', '5400',      '6000',     1,      0,      0,      'p.u.',     0.0198, 0.216, 0.025],
                ['L5401-5501', '5401',      '5501',     1,      0,      0,      'p.u.',     0.0105, 0.162, 0.08],
                ['L5401-5602', '5401',      '5602',     1,      0,      0,      'p.u.',     0.0096, 0.153, 0.09],
                ['L5401-6001', '5401',      '6001',     1,      0,      0,      'p.u.',     0.00384, 0.06, 0.028],
                ['L5402-6001', '5402',      '6001',     1,      0,      0,      'p.u.',     0.00042, 0.006, 0.003],
                ['L5500-5603', '5500',      '5603',     1,      0,      0,      'p.u.',     0.03, 0.36, 0.05],
                ['L5600-5601', '5600',      '5601',     1,      0,      0,      'p.u.',     0.018, 0.204, 0.02],
                ['L5600-5603', '5600',      '5603',     1,      0,      0,      'p.u.',     0.012, 0.132, 0.02],
                ['L5600-5620', '5600',      '5620',     1,      0,      0,      'p.u.',     0, 0.006, 0],
                ['L5600-6000', '5600',      '6000',     1,      0,      0,      'p.u.',     0.012, 0.12, 0.07],
                ['L5603-5610', '5603',      '5610',     1,      0,      0,      'p.u.',     0, 0.006, 0],
                ['L6000-6100', '6000',      '6100',     1,      0,      0,      'p.u.',     0.0204, 0.252, 0.03],
                ['L6500-6700-1', '6500',      '6700',     1,      0,      0,      'p.u.',     0.102, 1.08, 0.1],
                ['L6500-6700-2', '6500',      '6700',     1,      0,      0,      'p.u.',     0.06, 0.78, 0.12],
                ['L7000-7010', '7000',      '7010',     1,      0,      0,      'p.u.',     0, 0.006, 0],
                ['L7000-7020', '7000',      '7020',     1,      0,      0,      'p.u.',     0, 0.006, 0],
                ['L7000-7100-1', '7000',      '7100',     1,      0,      0,      'p.u.',     0.024, 0.072, 0.13],
                ['L7000-7100-2', '7000',      '7100',     1,      0,      0,      'p.u.',     0.024, 0.072, 0.13],
                ['L7000-7100-3', '7000',      '7100',     1,      0,      0,      'p.u.',     0.024, 0.084, 0.13],
                ['L8500-8600', '8500',      '8600',     1,      0,      0,      'p.u.',     0, 0.006, 0],
                ['L8500-8700', '8500',      '8700',     1,      0,      0,      'p.u.',     0, 0.006, 0],
            ],

        'transformers': [
                ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R',        'X',        'ratio'],
                ['T3244-3245', '3244', '3245', 1000,  0,          0,         0.005,     0.02,       1],
                ['T3701-3249', '3701', '3249', 1000,  0,          0,         0.02,      0.5,        1],
                ['T3359-3360', '3359', '3360', 1000,  0,          0,         0.005,     0.02,       1],
                ['T5101-5100', '5101', '5100', 1000,  0,          0,         0.0008,    0.0305,     1],
                ['T5300-5301', '5300', '5301', 1000,  0,          0,         0.0016,    0.061,      1],
                ['T5400-5401', '5400', '5401', 1000,  0,          0,         0.0032,    0.12,       1],
                ['T5400-5402', '5400', '5402', 1000,  0,          0,         0.0004,    0.015,      1],
                ['T5500-5501', '5500', '5501', 1000,  0,          0,         0.0004,    0.015,      1],
                ['T5601-6001', '5601', '6001', 1000,  0,          0,         0.0002,    0.0076,     1],
                ['T5603-5602', '5603', '5602', 1000,  0,          0,         0.0008,    0.0305,     1],
                ['T6000-6001', '6000', '6001', 1000,  0,          0,         0.0004,    0.015,      1],
                ['T6700-6701', '6700', '6701', 1000,  0,          0,         0.005,     0.02,       1],
            ],

        'loads': [
                ['name', 'bus', 'P', 'Q', 'model'],
                ['L3000-1', '3000', 1420.656, 567, 'Z'],
                ['L3000-2', '3000', 1420.656, 567, 'Z'],
                ['L3000-3', '3000', 1420.656, 567, 'Z'],
                ['L3020-1', '3020', 1219, 616, 'Z'],
                ['L3100-1', '3100', 621, 110, 'Z'],
                ['L3115-1', '3115', 621, 650, 'Z'],
                ['L3249-1', '3249', 2265, 650, 'Z'],
                ['L3300-1', '3300', 1217.358, 400, 'Z'],
                ['L3300-2', '3300', 1217.358, 400, 'Z'],
                ['L3359-1', '3359', 1460.829, 600, 'Z'],
                ['L3359-2', '3359', 1460.829, 600, 'Z'],
                ['L3359-3', '3359', 1460.829, 600, 'Z'],
                ['L3359-4', '3359', 1460.829, 600, 'Z'],
                ['L3360-1', '3360', -330, 262, 'Z'],
                ['L5100-1', '5100', 1154.17, 70, 'Z'],
                ['L5300-1', '5300', 2651, -70, 'Z'],
                ['L5400-1', '5400', 1149.765, 100, 'Z'],
                ['L5500-1', '5500', 2203.415, 200, 'Z'],
                ['L5500-2', '5500', 2203.415, 200, 'Z'],
                ['L5600-1', '5600', 674.862, 125, 'Z'],
                ['L5600-2', '5600', 674.862, 125, 'Z'],
                ['L5610-1', '5610', 1412, 363, 'Z'],
                ['L5620-1', '5620', 414, 175, 'Z'],
                ['L6100-1', '6100', 1199.755, 400, 'Z'],
                ['L6100-2', '6100', 1199.755, 400, 'Z'],
                ['L6500-1', '6500', 1013, 333, 'Z'],
                ['L6500-2', '6500', 1013, 333, 'Z'],
                ['L6500-3', '6500', 1013, 333, 'Z'],
                ['L6700-1', '6700', 2489, 150, 'Z'],
                ['L7000-1', '7000', 1593.526, 70, 'Z'],
                ['L7000-2', '7000', 1593.526, 70, 'Z'],
                ['L7000-3', '7000', 1593.526, 70, 'Z'],
                ['L7000-4', '7000', 1593.526, 70, 'Z'],
                ['L7000-5', '7000', 1593.526, 70, 'Z'],
                ['L7010-1', '7010', -1219, 600, 'Z'],
                ['L7020-1', '7020', 343, -4, 'Z'],
                ['L7100-1', '7100', 1431.684, 200, 'Z'],
                ['L7100-2', '7100', 1431.684, 200, 'Z'],
                ['L8500-1', '8500', 1240, 433, 'Z'],
                ['L8500-2', '8500', 1240, 433, 'Z'],
                ['L8500-3', '8500', 1240, 433, 'Z'],
                ['L8600-1', '8600', 546, 10, 'Z'],
                ['L8700-1', '8700', 628, 0, 'Z'],
            ],

        'shunts': [
                ['name', 'bus', 'Q', 'model'],
            ],

        # Note: All generators where X_q_t = X_q are salient pole generators. T_q0_t is set to one, but does not affect result.
        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',      'V_n', 'P',           'V',      'H',        'D',        'X_d',    'X_q',    'X_d_t', 'X_q_t',       'X_d_st',     'X_q_st', 'T_d0_t',  'T_q0_t',  'T_d0_st','T_q0_st'   ],
                ['G3000-1', '3000', 1300,       0,      1100,         1,        5.97,        0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    5,         1,         0.05,     0.05        ],
                ['G3000-2', '3000', 1300,       0,      1100,         1,        5.97,        0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    5,         1,         0.05,     0.05        ],
                ['G3000-3', '3000', 1300,       0,      0,            1,        5.97,        0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    5,         1,         0.05,     0.05        ],
                ['G3115-1', '3115', 1450.62,    0,      1175,         1,        4.741,       0,         0.946,    0.565,    0.29,    0.565,         0.23,         0.23,     7.57,      1,         0.045,    0.1         ],
                ['G3115-2', '3115', 1450.62,    0,      1175,         1,        4.741,       0,         0.946,    0.565,    0.29,    0.565,         0.23,         0.23,     7.57,      1,         0.045,    0.1         ],
                ['G3115-3', '3115', 1450.62,    0,      1175,         1,        4.741,       0,         0.946,    0.565,    0.29,    0.565,         0.23,         0.23,     7.57,      1,         0.045,    0.1         ],
                ['G3245-1', '3245', 1234.57,    0,      1000,         1,        3.3,         0,         0.75,     0.5,      0.25,    0.5,           0.15385,      0.15385,  5,         1,         0.06,     0.1         ],
                ['G3249-1', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3249-2', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3249-3', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3249-4', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3249-5', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3249-6', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3249-7', '3249', 1357,       0,      1042,         1,        4.543,       0,         1.036,    0.63,     0.28,    0.63,          0.21,         0.21,     10.13,     1,         0.06,     0.1         ],
                ['G3300-1', '3300', 1100,       0,      998.734,      1,        6,           0,         2.42,     2,        0.23,    0.4108,        0.16,         0.16,     10.8,      1,         0.05,     0.05        ],
                ['G3300-2', '3300', 1100,       0,      998.734,      1,        6,           0,         2.42,     2,        0.23,    0.4108,        0.16,         0.16,     10.8,      1,         0.05,     0.05        ],
                ['G3300-3', '3300', 1100,       0,      998.734,      1,        6,           0,         2.42,     2,        0.23,    0.4108,        0.16,         0.16,     10.8,      1,         0.05,     0.05        ],
                ['G3359-1', '3359', 1350,       0,      1110,         1,        4.82,        0,         2.13,     2.03,     0.31,    0.403,         0.1937,       0.1937,   4.75,      1,         0.05,     0.05        ],
                ['G3359-2', '3359', 1350,       0,      1100,         1,        4.82,        0,         2.13,     2.03,     0.31,    0.403,         0.1937,       0.1937,   4.75,      1,         0.05,     0.05        ],
                ['G3359-3', '3359', 1350,       0,      1100,         1,        4.82,        0,         2.13,     2.03,     0.31,    0.403,         0.1937,       0.1937,   4.75,      1,         0.05,     0.05        ],
                ['G3359-4', '3359', 1350,       0,      0,            1,        4.82,        0,         2.13,     2.03,     0.31,    0.403,         0.1937,       0.1937,   4.75,      1,         0.05,     0.05        ],
                ['G3359-5', '3359', 1350,       0,      0,            1,        4.82,        0,         2.13,     2.03,     0.31,    0.403,         0.1937,       0.1937,   4.75,      1,         0.05,     0.05        ],
                ['G3359-6', '3359', 1350,       0,      0,            1,        4.82,        0,         2.13,     2.03,     0.31,    0.403,         0.1937,       0.1937,   4.75,      1,         0.05,     0.05        ],
                ['G5100-1', '5100', 1200,       0,      972.437,      1,        3.9871,      0,         1.1332,   0.68315,  0.24302, 0.68315,       0.15135,      0.15135,  4.9629,    1,         0.05,     0.15        ],
                ['G5300-1', '5300', 1574.89,    0,      1275.661,     1,        3.5,         0,         1.14,     0.84,     0.34,    0.84,          0.26,         0.26,     6.4,       1,         0.05,     0.15        ],
                ['G5300-2', '5300', 1574.89,    0,      1275.661,     1,        3.5,         0,         1.14,     0.84,     0.34,    0.84,          0.26,         0.26,     6.4,       1,         0.05,     0.15        ],
                ['G5400-1', '5400', 1611.516,   0,      1305.328,     1.007,    4.1,         0,         1.02,     0.63,     0.25,    0.63,          0.16,         0.16,     6.5,       1,         0.05,     0.15        ],
                ['G5400-2', '5400', 1611.516,   0,      1305.328,     1.007,    4.1,         0,         1.02,     0.63,     0.25,    0.63,          0.16,         0.16,     6.5,       1,         0.05,     0.15        ],
                ['G5500-1', '5500', 1450,       0,      1131.563,     1.004,    3,           0,         1.2364,   0.65567,  0.37415, 0.65567,       0.22825,      0.22825,  7.198,     1,         0.05,     0.15        ],
                ['G5600-1', '5600', 1538.265,   0,      1245.995,     1.01,     3.5,         0,         1,        0.51325,  0.38,    0.51325,       0.28,         0.28,     7.85,      1,         0.05,     0.15        ],
                ['G5600-2', '5600', 1538.265,   0,      1245.995,     1.01,     3.5,         0,         1,        0.51325,  0.38,    0.51325,       0.28,         0.28,     7.85,      1,         0.05,     0.15        ],
                ['G6000-1', '6000', 896.59,     0,      735.73,       1.005,    3.5,         0,         1.28,     0.94,     0.37,    0.94,          0.28,         0.28,     9.7,       1,         0.05,     0.15        ],
                ['G6100-1', '6100', 1634.96,    0,      1329.061,     1,        3,           0,         1.2,      0.73,     0.37,    0.73,          0.18,         0.18,     9.9,       1,         0.05,     0.15        ],
                ['G6100-2', '6100', 1634.96,    0,      1329.061,     1,        3,           0,         1.2,      0.73,     0.37,    0.73,          0.18,         0.18,     9.9,       1,         0.05,     0.15        ],
                ['G6100-3', '6100', 1634.96,    0,      1329.061,     1,        3,           0,         1.2,      0.73,     0.37,    0.73,          0.18,         0.18,     9.9,       1,         0.05,     0.15        ],
                ['G6100-4', '6100', 1634.96,    0,      1329.061,     1,        3,           0,         1.2,      0.73,     0.37,    0.73,          0.18,         0.18,     9.9,       1,         0.05,     0.15        ],
                ['G6100-5', '6100', 1634.96,    0,      1329.061,     1,        3,           0,         1.2,      0.73,     0.37,    0.73,          0.18,         0.18,     9.9,       1,         0.05,     0.15        ],
                ['G6500-1', '6500', 1100,       0,      814.333,      1,        3.558,       0,         1.0679,   0.642,    0.23865, 0.642,         0.15802,      0.15802,  5.4855,    1,         0.05,     0.15        ],
                ['G6500-2', '6500', 1100,       0,      814.333,      1,        3.558,       0,         1.0679,   0.642,    0.23865, 0.642,         0.15802,      0.15802,  5.4855,    1,         0.05,     0.15        ],
                ['G6500-3', '6500', 1100,       0,      814.333,      1,        3.558,       0,         1.0679,   0.642,    0.23865, 0.642,         0.15802,      0.15802,  5.4855,    1,         0.05,     0.15        ],
                ['G6500-4', '6500', 1100,       0,      0,            1,        3.558,       0,         1.0679,   0.642,    0.23865, 0.642,         0.15802,      0.15802,  5.4855,    1,         0.05,     0.15        ],
                ['G6700-1', '6700', 2144.444,   0,      1753,         1.02,     3.592,       0,         1.1044,   0.66186,  0.25484, 0.66186,       0.17062,      0.17062,  5.24,      1,         0.05,     0.15        ],
                ['G6700-2', '6700', 2144.444,   0,      1753,         1.02,     3.592,       0,         1.1044,   0.66186,  0.25484, 0.66186,       0.17062,      0.17062,  5.24,      1,         0.05,     0.15        ],
                ['G7000-1', '7000', 1278,       0,      1085.5,       1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-2', '7000', 1278,       0,      1085.5,       1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-3', '7000', 1278,       0,      1085.5,       1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-4', '7000', 1278,       0,      1085.5,       1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-5', '7000', 1278,       0,      1085.5,       1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-6', '7000', 1278,       0,      1085.5,       1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-7', '7000', 1278,       0,      0,            1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-8', '7000', 1278,       0,      0,            1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7000-9', '7000', 1278,       0,      0,            1,        5.5,         0,         2.22,     2.13,     0.36,    0.468,         0.225,        0.225,    10,        1,         0.05,     0.05        ],
                ['G7100-1', '7100', 1000,       0,      715.333,      1,        3.2,         0,         0.75,     0.5,      0.25,    0.5,           0.15385,      0.15385,  5,         1,         0.06,     0.1         ],
                ['G7100-2', '7100', 1000,       0,      715.333,      1,        3.2,         0,         0.75,     0.5,      0.25,    0.5,           0.15385,      0.15385,  5,         1,         0.06,     0.1         ],
                ['G7100-3', '7100', 1000,       0,      715.333,      1,        3.2,         0,         0.75,     0.5,      0.25,    0.5,           0.15385,      0.15385,  5,         1,         0.06,     0.1         ],
                ['G8500-1', '8500', 1300,       0,      994,          1.02,     7,           0,         2.42,     2,        0.23,    0.4108,        0.17062,      0.17062,  10,        1,         0.05,     0.05        ],
                ['G8500-2', '8500', 1300,       0,      0,            1.02,     7,           0,         2.42,     2,        0.23,    0.4108,        0.17062,      0.17062,  10,        1,         0.05,     0.05        ],
                ['G8500-3', '8500', 1300,       0,      0,            1.02,     7,           0,         2.42,     2,        0.23,    0.4108,        0.17062,      0.17062,  10,        1,         0.05,     0.05        ],
                ['G8500-4', '8500', 1300,       0,      0,            1.02,     7,           0,         2.42,     2,        0.23,    0.4108,        0.17062,      0.17062,  10,        1,         0.05,     0.05        ],
                ['G8500-5', '8500', 1300,       0,      0,            1.02,     7,           0,         2.42,     2,        0.23,    0.4108,        0.17062,      0.17062,  10,        1,         0.05,     0.05        ],
                ['G8500-6', '8500', 1300,       0,      0,            1.02,     7,           0,         2.42,     2,        0.23,    0.4108,        0.17062,      0.17062,  10,        1,         0.05,     0.05        ],
            ]
        },
    }