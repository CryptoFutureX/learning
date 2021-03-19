import { useEffect, useState } from 'react'
import CoinLayer from '../api/CoinLayer'

const LiveDataLoader = () => {
    const [data,setData] = useState(null)

    useEffect(() => {
        const fetchData = async () => {
            const x = await CoinLayer.get('/live',{
                params:{
                    access_key:'07841d9d56dfeb8caddc8d5b853ed7cc',
                    target:'INR'
                }
            })

            const y = await CoinLayer.get('/live',{
                params:{
                    access_key:'07841d9d56dfeb8caddc8d5b853ed7cc',
                    target:'USD'
                }
            })
            console.log({inr:x.data.rates,usd:y.data.rates});
            setData({inr:x.data.rates,usd:y.data.rates})
        }

        fetchData()
    },[])

    if(data !== null)
    return data
}

export default LiveDataLoader