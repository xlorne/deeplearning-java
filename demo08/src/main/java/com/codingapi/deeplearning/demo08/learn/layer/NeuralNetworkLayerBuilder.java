package com.codingapi.deeplearning.demo08.learn.layer;

import lombok.extern.slf4j.Slf4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * 神经网络层构造器
 * @author lorne
 * @date 2019-10-31
 * @description
 */
@Slf4j
public class NeuralNetworkLayerBuilder implements Serializable {

    private boolean noOutLay = false;

    private List<NeuralNetworkLayer> layers;

    private NeuralNetworkLayerBuilder(){
        this.layers = new ArrayList<>();
    }

    private static NeuralNetworkLayerBuilder builder;

    public static NeuralNetworkLayerBuilder builder(){
        if(builder==null){
            builder = new NeuralNetworkLayerBuilder();
        }
        return builder;
    }


    public NeuralNetworkLayerBuilder addLayer(NeuralNetworkLayer layer){
        if(noOutLay){
            throw new RuntimeException("已经存在输出层");
        }
        int index = layers.size();
        layer.build(this,index);
        layers.add(layer);
        noOutLay = layer.isOutLayer();
        return this;
    }


    public List<NeuralNetworkLayer> list(){
        return layers;
    }

    public int size(){
        return layers.size();
    }

    public NeuralNetworkLayer get(int index) {
        return layers.get(index);
    }

    /**
     * 初始化所有层的权重 w,b
     */
    public void init(double lamdba,double alpha,long seed) {
        if(!noOutLay){
            throw new RuntimeException("没有输出层");
        }
        for (NeuralNetworkLayer layer:list()){
            layer.init(lamdba,alpha,seed);
        }
        log.info("init rand w,b ");
    }

    public NeuralNetworkLayerBuilder build(){
        return this;
    }
}
