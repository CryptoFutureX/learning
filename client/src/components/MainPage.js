import React, { useState } from 'react';
import './MainPage.css';
import Chart from 'react-apexcharts';
import options from './options';
import series from './sampleData';
import Menu from './Menu';
import { cryptoList, fiatList } from './DataList';

const MainPage = () => {
    const [selected, setSelected] = useState({
        Coin: 'Bitcoin',
        Fiat: 'USD',
    });

    const changeSelectedCoin = (e) => {
        setSelected({
            ...selected,
            [e.target.name]: e.target.value,
        });
    };

    return (
        <>
            <div className='GraphContainer'>
                <div className='crypto-menu'>
                    <Menu
                        selected={selected}
                        list={cryptoList}
                        changeSelectedCoin={changeSelectedCoin}
                        name='Coin'
                    />
                    <Menu
                        selected={selected}
                        list={fiatList}
                        changeSelectedCoin={changeSelectedCoin}
                        name='Fiat'
                    />
                </div>
                <div className='MainGraph'>
                    <Chart
                        options={options}
                        series={series['Bitcoin'][selected.Fiat]['series']}
                        type='candlestick'
                        height={600}
                    />
                </div>
            </div>
        </>
    );
};

export default MainPage;
