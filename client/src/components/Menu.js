import React from 'react';

const Menu = ({ selected, list, changeSelectedCoin, name }) => {
    return (
        <div class='dropdown2'>
            <p className='name'>{name} : </p>
            <button
                class='btn btn-secondary dropdown-toggle'
                type='button'
                id='dropdownMenuButton'
                data-toggle='dropdown'
                aria-haspopup='true'
                aria-expanded='false'
            >
                {selected[name]}
            </button>
            <div class='dropdown-menu' aria-labelledby='dropdownMenuButton'>
                {list.map((coin) => {
                    return (
                        <button
                            onClick={changeSelectedCoin}
                            class='dropdown-item'
                            name={name}
                            value={coin}
                        >
                            {coin}
                        </button>
                    );
                })}
            </div>
        </div>
    );
};

export default Menu;
