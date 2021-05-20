# This repository contains the learnings of the members of team CryptoFutureX

## Topics 
- D3Js : We initially decided to use D3Js for the frontend to create and plot prcice but soon realized that the D3Js is a low level library making it easier to customize according to the users but even the small changes takes a lot of effort to create and soon switched to ChartJs
- ChartJs : Next we stumbled on ChartJs since we initially decided to plot a line plot so it was rather easy since ChartJs is Simple, clean and engaging HTML5 based JavaScript charts.
- React ChartJs : Since ChartJs was mainly used for vanillaJs we needed a library which could configure and can be updated using React. This is where react-chart library was used
- React ApexCharts : Later we decided to give more investor oriented look to our webapp and decided to implement Candle Plot & Line Plot in our webapp.This is where ApexCharts Library came in handy   
- CryptoCompare : Onwards to the data fetching for the model training , prediction and plotting purpose. We used CryptoCompare lib which is python lib for fetching prices both latest as well as historical , coin listing across various exchanges and also trading using the api
- Multivariate Time Series Modelling using LSTM : For price prediction we created Multivariate Time Series Modelling using LSTM in tensorflow keras. Adjusting through different parameters we got an accuary of around ~81% for Bitcoin Data.


<!-- 
## Libs 
- [Pytorch](https://pytorch.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Transformers HuggingFace](https://github.com/huggingface/transformers) 
- [D3Js](https://d3js.org/) -->
