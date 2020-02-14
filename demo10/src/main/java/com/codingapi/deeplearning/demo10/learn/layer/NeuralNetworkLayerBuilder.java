package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.core.InputType;
import lombok.extern.slf4j.Slf4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
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
        if(layer instanceof FeedForwardLayer) {
            noOutLay = ((FeedForwardLayer)layer).isOutLayer();
        }
        return this;
    }


    public List<NeuralNetworkLayer> list(){
        return layers;
    }



    /**
     * 初始化所有层的权重 w,b
     */
    public void init(InputType inputType, double lamdba, double alpha, long seed) {
        if(!noOutLay){
            throw new RuntimeException("没有输出层");
        }
        LayerInitor layerInitor = new LayerInitor(lamdba,alpha,seed,inputType);
        List<NeuralNetworkLayer> list = list();
        log.info("init Weight .... ");
        for (int i=0;i<list.size();i++){
            NeuralNetworkLayer layer = list.get(i);
            log.info("layer:index:{}",i);
            log.info("input-type:{}",layerInitor.getInputType());
            layerInitor =  layer.initLayer(layerInitor);
            log.info("output-type:{}",layerInitor.getInputType());
        }

    }

    public NeuralNetworkLayerBuilder build(){
        return this;
    }

    public NeuralNetworkLayerIterator neuralNetworkLayerIterator(){
        return new NeuralNetworkLayerIterator(layers);
    }

    public FeedForwardLayer getNextFeedForwardLayer(int index) {
        if(index<0){
            return null;
        }
         NeuralNetworkLayer neuralNetworkLayer =  layers.get(index-1);
         if(neuralNetworkLayer instanceof FeedForwardLayer){
             return (FeedForwardLayer) neuralNetworkLayer;
         }
        return getNextFeedForwardLayer(index-1);
    }

    public FeedForwardLayer getAfterFeedForwardLayer(int index) {
        if(index>layers.size()){
            return null;
        }
        NeuralNetworkLayer neuralNetworkLayer =  layers.get(index+1);
        if(neuralNetworkLayer instanceof FeedForwardLayer){
            return (FeedForwardLayer) neuralNetworkLayer;
        }
        return getNextFeedForwardLayer(index+1);
    }

    public static class NeuralNetworkLayerIterator implements Iterator<NeuralNetworkLayer>{

        private List<NeuralNetworkLayer> layers;

        private int index = 0;

        private NeuralNetworkLayerIterator(List<NeuralNetworkLayer> layers) {
            this.layers = layers;
        }

        @Override
        public boolean hasNext() {
            if(index>=layers.size()){
                return false;
            }
            NeuralNetworkLayer layer = layers.get(index);
            if(layer != null){
                return true;
            }else{
                return false;
            }
        }

        @Override
        public NeuralNetworkLayer next() {
            return layers.get(index++);
        }
    }


    public FeedForwardLayerIterator feedForwardLayerIterator(){
        return new FeedForwardLayerIterator(layers);
    }

    public static  class  FeedForwardLayerIterator implements Iterator<FeedForwardLayer>{

        private List<NeuralNetworkLayer> layers;

        private int index = 0;

        private FeedForwardLayerIterator(List<NeuralNetworkLayer> layers) {
            this.layers = layers;
        }

        @Override
        public boolean hasNext() {
            if(index>=layers.size()){
               return false;
            }
            NeuralNetworkLayer layer = layers.get(index);
            if(layer instanceof FeedForwardLayer){
                return true;
            }
            int currentIndex = index;
            return hasNext(currentIndex++);
        }

        private boolean hasNext(int index) {
            if(index>=layers.size()){
                return false;
            }
            NeuralNetworkLayer layer = layers.get(index);
            if(layer instanceof FeedForwardLayer){
                return true;
            }
            return hasNext(index++);
        }

        @Override
        public FeedForwardLayer next() {
            NeuralNetworkLayer layer = layers.get(index++);
            if(layer instanceof FeedForwardLayer){
                return (FeedForwardLayer)layer;
            }else{
                return next();
            }
        }
    }

    public BFIterator bFIterator(){
        return new BFIterator(layers);
    }

    public static class BFIterator implements Iterator<FeedForwardLayer> {

        private  final List<NeuralNetworkLayer> layers;

        private int index;

        private final int count;

        public BFIterator(List<NeuralNetworkLayer> layers) {
            this.layers = layers;
            count = layers.size();
            index = count-1;
        }

        @Override
        public boolean hasNext() {
            if(index<0){
                return false;
            }
            NeuralNetworkLayer layer = layers.get(index);
            if(layer instanceof FeedForwardLayer){
                return true;
            }
            int currentIndex = index;
            return hasNext(currentIndex--);
        }

        private boolean hasNext(int index) {
            if(index<0){
                return false;
            }
            NeuralNetworkLayer layer = layers.get(index);
            if(layer instanceof FeedForwardLayer){
                return true;
            }
            return hasNext(index--);
        }


        @Override
        public FeedForwardLayer next() {
            NeuralNetworkLayer layer = layers.get(index--);
            if(layer instanceof FeedForwardLayer){
                return (FeedForwardLayer)layer;
            }else{
                return next();
            }
        }
    }
}
