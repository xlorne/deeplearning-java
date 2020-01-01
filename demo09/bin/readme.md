model.bin 
是当前项目训练后的模型，目前训练识别率为94.18%  
Coding:  
```
  @Test
    void train() throws IOException {

        int batchSize = 64;
        int rngSeed = 123;

        //保存的路径
        String file = "model.bin";

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        //创建神经网络层
        NeuralNetworkLayerBuilder neuralNetworkLayerBuilder
                = NeuralNetworkLayerBuilder.builder()
                .addLayer(DenseLayer.builder()
                        .input(28*28,1000)
                        .activation(new SigmoidActivation())
                        .build())
                .addLayer(DenseLayer.builder()
                        .input(1000,1000)
                        .activation(new SigmoidActivation())
                        .build())
                .addLayer(DenseLayer.builder()
                        .input(1000,10)
                        .activation(new SoftMaxActivation())
                        .isOutLayer(true)
                        .build())
                .build();

        //创建神经网络
        NeuralNetwork neuralNetwork =
                NeuralNetworkBuilder.builder()
                        .layers(neuralNetworkLayerBuilder)
                        .lossFunction(new LossNegativeLogLikelihood())
                        .seed(rngSeed)
                        .numEpochs(15)
                        .alpha(0.006)
                        .lambda(1e-4)
                        .build();

        //Loss函数监听
        neuralNetwork.initListeners(new ScoreLogTrainingListener(1));

        //训练数据
        neuralNetwork.fit(mnistTrain);

        //预测数据
        double count = 0;
        int success = 0;
        while (mnistTest.hasNext()){
            DataSet dataSet =  mnistTest.next();
            INDArray res = neuralNetwork.predict(dataSet.getFeatures());
            INDArray labels = dataSet.getLabels();
            int rows = res.rows();
            for(int i=0;i<rows;i++){
                count++;
                if(MaxUtils.maxIndex(res.getRow(i).toDoubleVector())==MaxUtils.maxIndex(labels.getRow(i).toDoubleVector())){
                    success++;
                }
            }
        }

        System.out.println(String.format("success:%f",(success/count)));

        //保存网络
        neuralNetwork.save(file);

    }


    @Test
    void test() throws IOException{

        int batchSize = 64;
        int rngSeed = 123;

        //保存的路径
        String file = "model.bin";

        NeuralNetwork neuralNetwork =  NeuralNetworkBuilder.load(file);

        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        //预测数据
        double count = 0;
        int success = 0;
        while (mnistTest.hasNext()){
            DataSet dataSet =  mnistTest.next();
            INDArray res = neuralNetwork.predict(dataSet.getFeatures());
            INDArray labels = dataSet.getLabels();
            int rows = res.rows();
            for(int i=0;i<rows;i++){
                count++;
                if(MaxUtils.maxIndex(res.getRow(i).toDoubleVector())==MaxUtils.maxIndex(labels.getRow(i).toDoubleVector())){
                    success++;
                }
            }
        }

        System.out.println(String.format("success:%f",(success/count)));


    }
```