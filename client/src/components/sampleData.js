const dateTime = (x) => {
    var y = new Date(x);
    y = `${y}`;
    console.log(y.slice(0, 15));
    return y.slice(0, 15);
};

const seriesData = {
    Bitcoin: {
        INR: {
            series: [
                {
                    data: [
                        {
                            x: dateTime('2021-03-12'),
                            y: [4549990.02, 5453529.12, 3721853.48, 4500000]
                        },
                        {
                            x: dateTime('2021-03-13'),
                            y: [4500000, 6582998.05, 3730832.85, 4408328.31]
                        },
                        {
                            x: dateTime('2021-03-14'),
                            y: [4408328.31, 5653870.07, 3945541.05, 4622801.46]
                        },
                        {
                            x: dateTime('2021-03-15'),
                            y: [4622801.46, 6270029.26, 3153997.06, 4520147.73]
                        },
                        {
                            x: dateTime('2021-03-16'),
                            y: [4520147.73, 5270925.57, 3349030.84, 4119226.9]
                        },
                        {
                            x: dateTime('2021-03-17'),
                            y: [4119226.9, 6308008.59, 3590468, 5677134.37]
                        },
                        {
                            x: dateTime('2021-03-18'),
                            y: [5677134.37, 5950008.03, 3882138.28, 4242346.91]
                        },
                        {
                            x: dateTime('2021-03-19'),
                            y: [4242346.91, 6265003.03, 3924107.76, 4432981.38]
                        },
                    ],
                },
            ],
        },
        USD: {
            series: [
                {
                    data: [
                        {
                            x: dateTime('2021-03-12'),
                            y: [57809.41, 58069.58, 55089.68, 57256.22],
                        },
                        {
                            x: dateTime('2021-03-13'),
                            y: [57256.22, 61749.15, 56103.32, 61179.79],
                        },
                        {
                            x: dateTime('2021-03-14'),
                            y: [61179.79, 61674.66, 58980.95, 58998.89]
                        },
                        {
                            x: dateTime('2021-03-15'),
                            y: [58998.89, 60589.17, 54857.68, 55665.01]
                        },
                        {
                            x: dateTime('2021-03-16'),
                            y: [55665.01, 56936.04, 53269.13, 56925.01]
                        },
                        {
                            x: dateTime('2021-03-17'),
                            y: [56925.01, 58975.28, 54156.11, 58909]
                        },
                        {
                            x: dateTime('2021-03-18'),
                            y: [58909, 60077.79, 57022.2, 57643.32]
                        },
                        {
                            x: dateTime('2021-03-19'),
                            y: [57643.32, 59237.76, 56283.37, 58427.47]
                        },
                    ],
                },
            ],
        },
        EUR: {
            series: [
                {
                    data: [
                        {
                            x: dateTime('2021-03-12'),
                            y: [57809.41, 58069.58, 55089.68, 57256.22],
                        },
                        {
                            x: dateTime('2021-03-13'),
                            y: [57256.22, 61749.15, 56103.32, 61179.79],
                        },
                        {
                            x: dateTime('2021-03-14'),
                            y: [61179.79, 61674.66, 58980.95, 58998.89]
                        },
                        {
                            x: dateTime('2021-03-15'),
                            y: [58998.89, 60589.17, 54857.68, 55665.01]
                        },
                        {
                            x: dateTime('2021-03-16'),
                            y: [55665.01, 56936.04, 53269.13, 56925.01]
                        },
                        {
                            x: dateTime('2021-03-17'),
                            y: [56925.01, 58975.28, 54156.11, 58909]
                        },
                        {
                            x: dateTime('2021-03-18'),
                            y: [58909, 60077.79, 57022.2, 57643.32]
                        },
                        {
                            x: dateTime('2021-03-19'),
                            y: [57643.32, 59237.76, 56283.37, 58427.47]
                        },
                    ],
                },
            ],
        },
        JPY: {
            series: [
                {
                    data: [
                        {
                            x: dateTime('2021-03-12'),
                            y: [6272698.69, 6298112.26, 6017582.98, 6234572.7]
                        },
                        {
                            x: dateTime('2021-03-13'),
                            y: [6234572.7, 6700124.65, 6123119.22, 6677376.03]
                        },
                        {
                            x: dateTime('2021-03-14'),
                            y: [6677376.03, 6725569.51, 6437387.32, 6441546.05]
                        },
                        {
                            x: dateTime('2021-03-15'),
                            y: [6441546.05, 6613435.07, 6031341.65, 6087252.16]
                        },
                        {
                            x: dateTime('2021-03-16'),
                            y: [6087252.16, 6202671.17, 5840128.37, 6201934.08]
                        },
                        {
                            x: dateTime('2021-03-17'),
                            y: [6201934.08, 6422272.47, 5936523.74, 6415370.02]
                        },
                        {
                            x: dateTime('2021-03-18'),
                            y: [6415370.02, 6544576.9, 6215445.87, 6284147.47]
                        },
                        {
                            x: dateTime('2021-03-19'),
                            y: [6284147.47, 6451416.95, 6133763.91, 6395757.19]
                        },
                    ],
                },
            ],
        },
    },
};

const series = [
    {
        data: [
            {
                x: dateTime('2021-03-08'),
                y: [50964.18, 52408.08, 49341.09, 52405.02],
            },
            {
                x: dateTime('2021-03-09'),
                y: [52405.02, 54933.93, 51888.98, 54928.7],
            },
            {
                x: dateTime('2021-03-10'),
                y: [54928.7, 57396.59, 53054.34, 55893.31],
            },
            {
                x: dateTime('2021-03-11'),
                y: [55893.31, 58142.77, 54314.65, 57809.41],
            },
            {
                x: dateTime('2021-03-12'),
                y: [57809.41, 58069.58, 55089.68, 57256.22],
            },
            {
                x: dateTime('2021-03-13'),
                y: [57256.22, 61749.15, 56103.32, 61179.79],
            },
            {
                x: dateTime('2021-03-14'),
                y: [61179.79, 61674.66, 58980.95, 58998.89],
            },
            {
                x: dateTime('2021-03-15'),
                y: [61179.79, 61674.66, 58980.95, 58998.89],
            },
        ],
    },
];

export default seriesData;