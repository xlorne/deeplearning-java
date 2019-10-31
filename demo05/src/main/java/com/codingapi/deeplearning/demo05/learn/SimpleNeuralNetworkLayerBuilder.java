package com.codingapi.deeplearning.demo05.learn;

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
public class SimpleNeuralNetworkLayerBuilder {

    private boolean noOutLay = true;

    private List<SimpleNeuralNetworkLayer> layers;

    private SimpleNeuralNetworkLayerBuilder(){
        this.layers = new ArrayList<>();
    }


    public static SimpleNeuralNetworkLayerBuilder build(){
        return  new SimpleNeuralNetworkLayerBuilder();
    }

    public SimpleNeuralNetworkLayerBuilder addLayer(int in,int out){
        if(!noOutLay){
            throw new RuntimeException("已经存在输出层");
        }
        int index = layers.size();
        layers.add(new SimpleNeuralNetworkLayer(layers,in,out,index,false));
        return this;
    }

    public SimpleNeuralNetworkLayerBuilder outLayer(int in,int out){
        if(!noOutLay){
            throw new RuntimeException("已经存在输出层");
        }
        int index = layers.size();
        layers.add(new SimpleNeuralNetworkLayer(layers,in,out,index,true));
        noOutLay = false;
        return this;
    }

    public List<SimpleNeuralNetworkLayer> list(){
        return layers;
    }

    public int size(){
        return layers.size();
    }

    public SimpleNeuralNetworkLayer get(int index) {
        return layers.get(index);
    }

    /**
     * 初始化所有层的权重 w,b
     */
    public void init() {
        if(noOutLay){
            throw new RuntimeException("没有输出层");
        }
        for (SimpleNeuralNetworkLayer layer:list()){
            layer.init();
        }
        log.info("init rand w,b ");
    }
}
