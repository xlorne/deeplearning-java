package com.codingapi.deeplearning.demo06.learn;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * 神经网络层构造器
 * @author lorne
 * @date 2019-10-31
 * @description
 */
@Slf4j
public class NeuralNetworkLayerBuilder {

    private boolean noOutLay = false;

    private List<NeuralNetworkLayer> layers;

    private NeuralNetworkLayerBuilder(){
        this.layers = new ArrayList<>();
    }


    public static NeuralNetworkLayerBuilder Builder(){
        return  new NeuralNetworkLayerBuilder();
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
    public void init() {
        if(!noOutLay){
            throw new RuntimeException("没有输出层");
        }
        for (NeuralNetworkLayer layer:list()){
            layer.init();
        }
        log.info("init rand w,b ");
    }

    public NeuralNetworkLayerBuilder builder(){
        return this;
    }
}
