package com.codingapi.deeplearning.demo10.learn.layer;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public abstract class BaseLayer implements FeedForwardLayer{


    //当前层数索引
    protected int index;
    //所有的网络层
    protected NeuralNetworkLayerBuilder layerBuilder;


    @Override
    public void build(NeuralNetworkLayerBuilder layerBuilder, int index) {
        this.index = index;
        this.layerBuilder = layerBuilder;
    }

    /**
     * 获取index + 1 FeedForwardLayer
     * @return
     */
    protected FeedForwardLayer getAfterFeedForwardLayer(){
        return layerBuilder.getAfterFeedForwardLayer(index);
    }

    /**
     * 获取index - 1 FeedForwardLayer
     * @return
     */
    protected NeuralNetworkLayer getNextFeedForwardLayer(){
        return layerBuilder.getNextFeedForwardLayer(index);
    }
}
