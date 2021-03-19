import { useEffect, useState } from 'react';
import CoinLayer from '../api/CoinLayer';

const TimeFrameData = () => {
    const [data, setData] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            const x = await CoinLayer.get('/timeframe', {
                params: {
                    access_key: '07841d9d56dfeb8caddc8d5b853ed7cc',
                    target: 'INR',
                    symbols: 'BTC,ETH',
                    start_date:'2018-04-01',
                    end_date:'2018-04-30'
                },
            });
            console.log(x.data);
            setData(x.data);
        };
        fetchData();
    }, []);

    if (data !== null) return data;
};

export default TimeFrameData;
