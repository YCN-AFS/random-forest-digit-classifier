import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// Import Deeplearning4j classes
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

// Import Weka classes for data loading
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import javax.imageio.ImageIO;

/**
 * Ứng dụng GUI để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST CSV
 * Sử dụng Convolutional Neural Network (CNN) với Deeplearning4j
 * 
 * @author AI Assistant
 * @version 2.0
 */
public class MLGuiAppCNN extends JFrame {
    
    // Các thành phần giao diện
    private JButton btnLoadTrain, btnLoadTest, btnTrain, btnEvaluate, btnPredict, btnShowCharts, btnSaveModel, btnLoadModel, btnSaveResults, btnSaveCharts, btnSaveAll;
    private JTextArea txtResults;
    private JTextField txtTestRow;
    private JLabel lblStatus;
    private JProgressBar progressBar;
    private JLabel lblProgress;
    private JTabbedPane tabbedPane;
    private JPanel chartPanel;
    
    // Dữ liệu và mô hình
    private Instances trainData, testData;
    private MultiLayerNetwork model;
    private boolean isModelTrained = false;
    private Evaluation lastEvaluation;
    private DataSet trainDataSet, testDataSet;
    
    // Đường dẫn file
    private String trainFilePath = "";
    private String testFilePath = "";
    
    // Cấu hình CNN
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int CHANNELS = 1;
    private static final int NUM_CLASSES = 10;
    private static final int BATCH_SIZE = 64;
    private static final int EPOCHS = 10;
    
    public MLGuiAppCNN() {
        initializeGUI();
    }
    
    private void initializeGUI() {
        setTitle("MNIST Handwritten Digit Recognition - CNN Version");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 800);
        setLocationRelativeTo(null);
        
        // Tạo layout chính
        setLayout(new BorderLayout());
        
        // Tạo control panel
        JPanel controlPanel = createControlPanel();
        add(controlPanel, BorderLayout.NORTH);
        
        // Tạo tabbed pane cho kết quả và biểu đồ
        tabbedPane = new JTabbedPane();
        
        // Tab kết quả
        JPanel resultPanel = createResultPanel();
        tabbedPane.addTab("Kết quả", resultPanel);
        
        // Tab biểu đồ
        chartPanel = new JPanel(new BorderLayout());
        tabbedPane.addTab("Biểu đồ", chartPanel);
        
        add(tabbedPane, BorderLayout.CENTER);
        
        // Status bar
        JPanel statusPanel = createStatusPanel();
        add(statusPanel, BorderLayout.SOUTH);
    }
    
    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new GridLayout(3, 3, 5, 5));
        
        // Hàng 1
        btnLoadTrain = new JButton("Chọn file Train");
        btnLoadTrain.addActionListener(e -> loadTrainData());
        panel.add(btnLoadTrain);
        
        btnLoadTest = new JButton("Chọn file Test");
        btnLoadTest.addActionListener(e -> loadTestData());
        panel.add(btnLoadTest);
        
        btnTrain = new JButton("Huấn luyện CNN");
        btnTrain.addActionListener(e -> trainModel());
        btnTrain.setEnabled(false);
        panel.add(btnTrain);
        
        // Hàng 2
        btnEvaluate = new JButton("Đánh giá");
        btnEvaluate.addActionListener(e -> evaluateModel());
        btnEvaluate.setEnabled(false);
        panel.add(btnEvaluate);
        
        btnShowCharts = new JButton("Hiển thị biểu đồ");
        btnShowCharts.addActionListener(e -> showCharts());
        btnShowCharts.setEnabled(false);
        panel.add(btnShowCharts);
        
        btnPredict = new JButton("Dự đoán");
        btnPredict.addActionListener(e -> predictDigit());
        btnPredict.setEnabled(false);
        panel.add(btnPredict);
        
        // Hàng 3
        btnSaveModel = new JButton("Lưu mô hình");
        btnSaveModel.addActionListener(e -> saveModel());
        btnSaveModel.setEnabled(false);
        panel.add(btnSaveModel);
        
        btnLoadModel = new JButton("Load mô hình");
        btnLoadModel.addActionListener(e -> loadModel());
        panel.add(btnLoadModel);
        
        btnSaveResults = new JButton("Lưu kết quả");
        btnSaveResults.addActionListener(e -> saveResults());
        btnSaveResults.setEnabled(false);
        panel.add(btnSaveResults);
        
        // Hàng 4
        btnSaveCharts = new JButton("Lưu biểu đồ");
        btnSaveCharts.addActionListener(e -> saveCharts());
        btnSaveCharts.setEnabled(false);
        panel.add(btnSaveCharts);
        
        btnSaveAll = new JButton("Lưu tất cả");
        btnSaveAll.addActionListener(e -> saveAll());
        btnSaveAll.setEnabled(false);
        panel.add(btnSaveAll);
        
        // Empty label để fill grid
        panel.add(new JLabel());
        
        return panel;
    }
    
    private JPanel createResultPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        txtResults = new JTextArea(20, 50);
        txtResults.setEditable(false);
        txtResults.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        
        JScrollPane scrollPane = new JScrollPane(txtResults);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        // Panel nhập dữ liệu test
        JPanel inputPanel = new JPanel(new FlowLayout());
        inputPanel.add(new JLabel("Chọn dòng test (0-" + (testData != null ? testData.numInstances() - 1 : "?") + "):"));
        txtTestRow = new JTextField(10);
        inputPanel.add(txtTestRow);
        inputPanel.add(btnPredict);
        
        panel.add(inputPanel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    private JPanel createStatusPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        lblStatus = new JLabel("Sẵn sàng");
        panel.add(lblStatus, BorderLayout.WEST);
        
        progressBar = new JProgressBar();
        progressBar.setStringPainted(true);
        panel.add(progressBar, BorderLayout.CENTER);
        
        lblProgress = new JLabel("");
        panel.add(lblProgress, BorderLayout.EAST);
        
        return panel;
    }
    
    private void loadTrainData() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("CSV Files", "csv"));
        fileChooser.setCurrentDirectory(new File("."));
        
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            trainFilePath = fileChooser.getSelectedFile().getAbsolutePath();
            loadData(trainFilePath, true);
        }
    }
    
    private void loadTestData() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("CSV Files", "csv"));
        fileChooser.setCurrentDirectory(new File("."));
        
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            testFilePath = fileChooser.getSelectedFile().getAbsolutePath();
            loadData(testFilePath, false);
        }
    }
    
    private void loadData(String filePath, boolean isTraining) {
        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                publish("🔄 Đang load dữ liệu " + (isTraining ? "training" : "test") + "...");
                
                try {
                    // Load CSV data using Weka
                    CSVLoader loader = new CSVLoader();
                    loader.setSource(new File(filePath));
                    Instances data = loader.getDataSet();
                    
                    // Set class attribute (first column - label)
                    data.setClassIndex(0);
                    
                    // Convert label from numeric to nominal
                    NumericToNominal convertLabel = new NumericToNominal();
                    convertLabel.setAttributeIndices("first");
                    convertLabel.setInputFormat(data);
                    data = Filter.useFilter(data, convertLabel);
                    
                    // Normalize pixel values to 0-1 range
                    Normalize normalize = new Normalize();
                    normalize.setInputFormat(data);
                    data = Filter.useFilter(data, normalize);
                    
                    // Set class index again after filtering
                    data.setClassIndex(0);
                    
                    if (isTraining) {
                        trainData = data;
                        trainDataSet = convertToDataSet(data, true);
                        publish("✅ Đã load dữ liệu training: " + data.numInstances() + " mẫu");
                        publish("   - Số thuộc tính: " + data.numAttributes());
                        publish("   - Class attribute: " + data.classAttribute().name());
                        publish("   - Số lớp: " + data.numClasses());
                    } else {
                        testData = data;
                        testDataSet = convertToDataSet(data, false);
                        publish("✅ Đã load dữ liệu test: " + data.numInstances() + " mẫu");
                        publish("   - Số thuộc tính: " + data.numAttributes());
                        publish("   - Class attribute: " + data.classAttribute().name());
                        publish("   - Số lớp: " + data.numClasses());
                    }
                    
                    // Enable train button if both datasets are loaded
                    if (trainData != null && testData != null) {
                        SwingUtilities.invokeLater(() -> {
                            btnTrain.setEnabled(true);
                            lblStatus.setText("Dữ liệu đã sẵn sàng để huấn luyện");
                        });
                    }
                    
                } catch (Exception e) {
                    publish("❌ Lỗi load dữ liệu: " + e.getMessage());
                    e.printStackTrace();
                }
                
                return null;
            }
            
            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    txtResults.append(message + "\n");
                }
                txtResults.setCaretPosition(txtResults.getDocument().getLength());
            }
        };
        
        worker.execute();
    }
    
    private DataSet convertToDataSet(Instances data, boolean isTraining) {
        try {
            int numSamples = data.numInstances();
            int numFeatures = data.numAttributes() - 1; // Exclude class attribute
            
            // Create feature matrix (samples x features)
            INDArray features = Nd4j.zeros(numSamples, CHANNELS, HEIGHT, WIDTH);
            INDArray labels = Nd4j.zeros(numSamples, NUM_CLASSES);
            
            for (int i = 0; i < numSamples; i++) {
                Instance instance = data.instance(i);
                
                // Convert pixel values to 28x28 image
                for (int row = 0; row < HEIGHT; row++) {
                    for (int col = 0; col < WIDTH; col++) {
                        int pixelIndex = row * WIDTH + col + 1; // +1 to skip class attribute
                        if (pixelIndex < instance.numAttributes()) {
                            double pixelValue = instance.value(pixelIndex);
                            features.putScalar(i, 0, row, col, pixelValue);
                        }
                    }
                }
                
                // Convert class label to one-hot encoding
                int classIndex = (int) instance.classValue();
                labels.putScalar(i, classIndex, 1.0);
            }
            
            return new DataSet(features, labels);
            
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    private void trainModel() {
        if (trainData == null || testData == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng load dữ liệu training và test trước!");
            return;
        }
        
        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                publish("🔄 Bắt đầu huấn luyện CNN...");
                publish("📊 Thông tin dữ liệu:");
                publish("   - Số mẫu training: " + trainData.numInstances());
                publish("   - Số mẫu test: " + testData.numInstances());
                publish("   - Kích thước ảnh: " + HEIGHT + "x" + WIDTH);
                publish("   - Số lớp: " + NUM_CLASSES);
                
                try {
                    // Create CNN configuration
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Adam(0.001))
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(CHANNELS)
                            .stride(1, 1)
                            .nOut(20)
                            .activation(Activation.RELU)
                            .build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .stride(1, 1)
                            .nOut(50)
                            .activation(Activation.RELU)
                            .build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                        .layer(4, new DenseLayer.Builder()
                            .activation(Activation.RELU)
                            .nOut(500)
                            .build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(NUM_CLASSES)
                            .activation(Activation.SOFTMAX)
                            .build())
                        .build();
                    
                    // Create and initialize model
                    model = new MultiLayerNetwork(conf);
                    model.init();
                    model.setListeners(new ScoreIterationListener(100));
                    
                    publish("🌲 Bắt đầu huấn luyện CNN...");
                    publish("⏳ Quá trình này có thể mất vài phút...");
                    
                    long startTime = System.currentTimeMillis();
                    
                    // Train the model
                    for (int epoch = 0; epoch < EPOCHS; epoch++) {
                        publish("📈 Epoch " + (epoch + 1) + "/" + EPOCHS);
                        model.fit(trainDataSet);
                        
                        // Evaluate on test set
                        INDArray predictions = model.output(testDataSet.getFeatures());
                        Evaluation eval = model.evaluate(testDataSet);
                        double accuracy = eval.accuracy();
                        publish("   Accuracy: " + String.format("%.2f", accuracy * 100) + "%");
                    }
                    
                    long endTime = System.currentTimeMillis();
                    double trainingTime = (endTime - startTime) / 1000.0;
                    
                    publish("✅ Hoàn thành huấn luyện CNN!");
                    publish("⏱️ Thời gian: " + String.format("%.2f", trainingTime) + " giây");
                    publish("🎯 Mô hình sẵn sàng để dự đoán!");
                    
                    // Enable buttons
                    SwingUtilities.invokeLater(() -> {
                        isModelTrained = true;
                        btnEvaluate.setEnabled(true);
                        btnPredict.setEnabled(true);
                        btnSaveModel.setEnabled(true);
                        lblStatus.setText("Mô hình CNN đã được huấn luyện");
                    });
                    
                } catch (Exception e) {
                    publish("❌ Lỗi huấn luyện: " + e.getMessage());
                    e.printStackTrace();
                }
                
                return null;
            }
            
            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    txtResults.append(message + "\n");
                }
                txtResults.setCaretPosition(txtResults.getDocument().getLength());
            }
        };
        
        worker.execute();
    }
    
    private void evaluateModel() {
        if (!isModelTrained || model == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng huấn luyện mô hình trước!");
            return;
        }
        
        if (testData == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng load dữ liệu test trước!");
            return;
        }
        
        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                publish("🔄 Đang đánh giá mô hình CNN...");
                
                try {
                    // Evaluate model
                    Evaluation eval = model.evaluate(testDataSet);
                    double accuracy = eval.accuracy();
                    
                    publish("📊 KẾT QUẢ ĐÁNH GIÁ:");
                    publish("========================");
                    publish("Accuracy: " + String.format("%.2f", accuracy * 100) + "%");
                    publish("Số mẫu đúng: " + (int)(accuracy * testData.numInstances()) + "/" + testData.numInstances());
                    publish("Số mẫu sai: " + (int)((1 - accuracy) * testData.numInstances()));
                    publish("");
                    
                    // Per-class accuracy
                    publish("📋 CHI TIẾT TỪNG LỚP:");
                    publish("====================");
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        double classAccuracy = eval.f1(i);
                        publish("Chữ số " + i + ": " + String.format("%.2f", classAccuracy * 100) + "%");
                    }
                    
                    // Store evaluation for charts
                    lastEvaluation = eval;
                    
                    // Enable save buttons
                    SwingUtilities.invokeLater(() -> {
                        btnSaveResults.setEnabled(true);
                        btnSaveCharts.setEnabled(true);
                        btnSaveAll.setEnabled(true);
                        btnShowCharts.setEnabled(true);
                        lblStatus.setText("Đánh giá hoàn thành - Accuracy: " + String.format("%.2f", accuracy * 100) + "%");
                    });
                    
                } catch (Exception e) {
                    publish("❌ Lỗi đánh giá: " + e.getMessage());
                    e.printStackTrace();
                }
                
                return null;
            }
            
            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    txtResults.append(message + "\n");
                }
                txtResults.setCaretPosition(txtResults.getDocument().getLength());
            }
        };
        
        worker.execute();
    }
    
    private void predictDigit() {
        if (!isModelTrained || model == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng huấn luyện mô hình trước!");
            return;
        }
        
        if (testData == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng load dữ liệu test trước!");
            return;
        }
        
        try {
            int rowIndex = Integer.parseInt(txtTestRow.getText().trim());
            if (rowIndex < 0 || rowIndex >= testData.numInstances()) {
                JOptionPane.showMessageDialog(this, "Dòng không hợp lệ! Vui lòng nhập từ 0 đến " + (testData.numInstances() - 1));
                return;
            }
            
            // Get test instance
            Instance testInstance = testData.instance(rowIndex);
            int actualLabel = (int) testInstance.classValue();
            
            // Convert to DataSet for prediction
            DataSet singleTest = convertToDataSet(testData, false);
            INDArray features = singleTest.getFeatures().getRow(rowIndex).reshape(1, CHANNELS, HEIGHT, WIDTH);
            
            // Make prediction
            INDArray prediction = model.output(features);
            int predictedLabel = prediction.argMax(1).getInt(0);
            double confidence = prediction.getDouble(0, predictedLabel);
            
            // Display result
            txtResults.append("\n🔮 DỰ ĐOÁN CHỮ SỐ:\n");
            txtResults.append("==================\n");
            txtResults.append("Dòng test: " + rowIndex + "\n");
            txtResults.append("Chữ số thực tế: " + actualLabel + "\n");
            txtResults.append("Chữ số dự đoán: " + predictedLabel + "\n");
            txtResults.append("Độ tin cậy: " + String.format("%.2f", confidence * 100) + "%\n");
            txtResults.append("Kết quả: " + (actualLabel == predictedLabel ? "✅ ĐÚNG" : "❌ SAI") + "\n\n");
            
            txtResults.setCaretPosition(txtResults.getDocument().getLength());
            
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Vui lòng nhập số hợp lệ!");
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Lỗi dự đoán: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void showCharts() {
        if (lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng đánh giá mô hình trước!");
            return;
        }
        
        chartPanel.removeAll();
        
        // Create confusion matrix chart
        JPanel confusionMatrixPanel = createConfusionMatrixChart();
        chartPanel.add(confusionMatrixPanel, BorderLayout.CENTER);
        
        chartPanel.revalidate();
        chartPanel.repaint();
    }
    
    private JPanel createConfusionMatrixChart() {
        JPanel panel = new JPanel(new BorderLayout());
        
        if (lastEvaluation == null) {
            JLabel noDataLabel = new JLabel("Chưa có dữ liệu để vẽ biểu đồ", JLabel.CENTER);
            noDataLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
            panel.add(noDataLabel, BorderLayout.CENTER);
            return panel;
        }
        
        try {
            // Get confusion matrix
            double[][] confusionMatrix = lastEvaluation.confusionMatrix();
            
            if (confusionMatrix == null || confusionMatrix.length == 0) {
                JLabel noDataLabel = new JLabel("Không có dữ liệu confusion matrix", JLabel.CENTER);
                noDataLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
                panel.add(noDataLabel, BorderLayout.CENTER);
                return panel;
            }
            
            // Create confusion matrix visualization
            JPanel matrixPanel = new JPanel(new GridLayout(confusionMatrix.length + 1, confusionMatrix[0].length + 1, 2, 2));
            matrixPanel.setBorder(BorderFactory.createTitledBorder("Confusion Matrix - CNN Model"));
            
            // Add headers
            matrixPanel.add(new JLabel("")); // Empty corner
            for (int j = 0; j < confusionMatrix[0].length; j++) {
                JLabel headerLabel = new JLabel("Pred " + j, JLabel.CENTER);
                headerLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 10));
                matrixPanel.add(headerLabel);
            }
            
            // Add rows
            for (int i = 0; i < confusionMatrix.length; i++) {
                JLabel rowLabel = new JLabel("Actual " + i, JLabel.CENTER);
                rowLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 10));
                matrixPanel.add(rowLabel);
                
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    JLabel cellLabel = new JLabel(String.valueOf((int) confusionMatrix[i][j]), JLabel.CENTER);
                    cellLabel.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));
                    cellLabel.setOpaque(true);
                    
                    // Color coding: green for correct predictions, red for incorrect
                    if (i == j) {
                        cellLabel.setBackground(new Color(144, 238, 144)); // Light green
                    } else {
                        cellLabel.setBackground(new Color(255, 182, 193)); // Light red
                    }
                    
                    matrixPanel.add(cellLabel);
                }
            }
            
            panel.add(matrixPanel, BorderLayout.CENTER);
            
        } catch (Exception e) {
            JLabel errorLabel = new JLabel("Lỗi tạo biểu đồ: " + e.getMessage(), JLabel.CENTER);
            errorLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
            panel.add(errorLabel, BorderLayout.CENTER);
        }
        
        return panel;
    }
    
    private void saveModel() {
        if (!isModelTrained || model == null) {
            JOptionPane.showMessageDialog(this, "Không có mô hình để lưu!");
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("Model Files", "model"));
        fileChooser.setSelectedFile(new File("cnn_model_" + System.currentTimeMillis() + ".model"));
        
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File modelFile = fileChooser.getSelectedFile();
                model.save(modelFile);
                JOptionPane.showMessageDialog(this, "Đã lưu mô hình CNN thành công!\nFile: " + modelFile.getName());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi lưu mô hình: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private void loadModel() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("Model Files", "model"));
        
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File modelFile = fileChooser.getSelectedFile();
                model = MultiLayerNetwork.load(modelFile, true);
                isModelTrained = true;
                
                btnEvaluate.setEnabled(true);
                btnPredict.setEnabled(true);
                btnSaveModel.setEnabled(true);
                
                JOptionPane.showMessageDialog(this, "Đã load mô hình CNN thành công!\nFile: " + modelFile.getName());
                lblStatus.setText("Mô hình CNN đã được load");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi load mô hình: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private void saveResults() {
        if (lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Không có kết quả để lưu!");
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("Text Files", "txt"));
        fileChooser.setSelectedFile(new File("cnn_results_" + System.currentTimeMillis() + ".txt"));
        
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File resultFile = fileChooser.getSelectedFile();
                try (PrintWriter writer = new PrintWriter(new FileWriter(resultFile))) {
                    writer.println("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CNN");
                    writer.println("=============================");
                    writer.println("Thời gian: " + new java.util.Date());
                    writer.println("Accuracy: " + String.format("%.2f", lastEvaluation.pctCorrect()) + "%");
                    writer.println("Số mẫu: " + testData.numInstances());
                    writer.println();
                    writer.println("Confusion Matrix:");
                    writer.println(lastEvaluation.toMatrixString());
                }
                
                JOptionPane.showMessageDialog(this, "Đã lưu kết quả thành công!\nFile: " + resultFile.getName());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi lưu kết quả: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private void saveCharts() {
        if (lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Không có biểu đồ để lưu!");
            return;
        }
        
        JFileChooser dirChooser = new JFileChooser();
        dirChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        dirChooser.setDialogTitle("Chọn thư mục lưu biểu đồ");
        
        if (dirChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File dir = dirChooser.getSelectedFile();
                String timestamp = String.valueOf(System.currentTimeMillis());
                
                // Save confusion matrix
                JPanel confusionPanel = createConfusionMatrixChart();
                BufferedImage confusionImage = createImageFromPanel(confusionPanel);
                File confusionFile = new File(dir, "cnn_confusion_matrix_" + timestamp + ".png");
                ImageIO.write(confusionImage, "PNG", confusionFile);
                
                JOptionPane.showMessageDialog(this, "Đã lưu biểu đồ thành công!\nThư mục: " + dir.getName());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi lưu biểu đồ: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private void saveAll() {
        if (!isModelTrained || model == null || lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Không có dữ liệu để lưu!");
            return;
        }
        
        JFileChooser dirChooser = new JFileChooser();
        dirChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        dirChooser.setDialogTitle("Chọn thư mục lưu tất cả");
        
        if (dirChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File dir = dirChooser.getSelectedFile();
                String timestamp = String.valueOf(System.currentTimeMillis());
                
                // Save model
                File modelFile = new File(dir, "cnn_model_" + timestamp + ".model");
                model.save(modelFile);
                
                // Save results
                File resultFile = new File(dir, "cnn_results_" + timestamp + ".txt");
                try (PrintWriter writer = new PrintWriter(new FileWriter(resultFile))) {
                    writer.println("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CNN");
                    writer.println("=============================");
                    writer.println("Thời gian: " + new java.util.Date());
                    writer.println("Accuracy: " + String.format("%.2f", lastEvaluation.pctCorrect()) + "%");
                    writer.println("Số mẫu: " + testData.numInstances());
                    writer.println();
                    writer.println("Confusion Matrix:");
                    writer.println(lastEvaluation.toMatrixString());
                }
                
                // Save charts
                JPanel confusionPanel = createConfusionMatrixChart();
                BufferedImage confusionImage = createImageFromPanel(confusionPanel);
                File confusionFile = new File(dir, "cnn_confusion_matrix_" + timestamp + ".png");
                ImageIO.write(confusionImage, "PNG", confusionFile);
                
                JOptionPane.showMessageDialog(this, "Đã lưu tất cả thành công!\nThư mục: " + dir.getName());
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi lưu tất cả: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
    
    private BufferedImage createImageFromPanel(JPanel panel) {
        // Ensure panel is properly sized
        panel.setSize(panel.getPreferredSize());
        panel.doLayout();
        
        int width = panel.getWidth();
        int height = panel.getHeight();
        
        if (width <= 0 || height <= 0) {
            width = 800;
            height = 600;
        }
        
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, width, height);
        
        // Create a temporary frame to ensure proper rendering
        JFrame tempFrame = new JFrame();
        tempFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        tempFrame.add(panel);
        tempFrame.pack();
        tempFrame.setVisible(false);
        
        try {
            Thread.sleep(200); // Allow time for rendering
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        panel.paint(g2d);
        g2d.dispose();
        tempFrame.dispose();
        
        return image;
    }
    
    public static void main(String[] args) {
        // Set system look and feel
        try {
            UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeel());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SwingUtilities.invokeLater(() -> {
            new MLGuiAppCNN().setVisible(true);
        });
    }
}