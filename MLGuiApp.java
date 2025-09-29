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

// Import Weka classes
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

/**
 * Ứng dụng GUI để nhận dạng chữ số viết tay từ bộ dữ liệu MNIST CSV
 * Sử dụng Weka RandomForest classifier
 * 
 * @author AI Assistant
 * @version 1.0
 */
public class MLGuiApp extends JFrame {
    
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
    private Classifier model;
    private boolean isModelTrained = false;
    private Evaluation lastEvaluation;
    
    // Đường dẫn file
    private String trainFilePath = "";
    private String testFilePath = "";
    
    public MLGuiApp() {
        initializeGUI();
    }
    
    /**
     * Khởi tạo giao diện người dùng
     */
    private void initializeGUI() {
        setTitle("MNIST Handwritten Digit Recognition - ML GUI App");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        // Tạo tabbed pane
        tabbedPane = new JTabbedPane();
        
        // Tab 1: Kết quả text
        JPanel resultPanel = createResultPanel();
        tabbedPane.addTab("Kết quả", resultPanel);
        
        // Tab 2: Biểu đồ
        chartPanel = createChartPanel();
        tabbedPane.addTab("Biểu đồ", chartPanel);
        
        // Panel chính
        JPanel mainPanel = new JPanel(new BorderLayout());
        
        // Panel điều khiển (phía trên)
        JPanel controlPanel = createControlPanel();
        mainPanel.add(controlPanel, BorderLayout.NORTH);
        
        // Tabbed pane (giữa)
        mainPanel.add(tabbedPane, BorderLayout.CENTER);
        
        // Panel dự đoán (phía dưới)
        JPanel predictPanel = createPredictPanel();
        mainPanel.add(predictPanel, BorderLayout.SOUTH);
        
        add(mainPanel);
        
        // Thiết lập kích thước và vị trí
        setSize(800, 700);
        setLocationRelativeTo(null);
        setVisible(true);
    }
    
    /**
     * Tạo panel điều khiển với các nút chức năng
     */
    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new GridLayout(3, 3, 5, 5)); // 3 hàng, 3 cột, khoảng cách 5px
        panel.setBorder(BorderFactory.createTitledBorder("Điều khiển"));
        
        btnLoadTrain = new JButton("Chọn file Train");
        btnLoadTest = new JButton("Chọn file Test");
        btnTrain = new JButton("Huấn luyện mô hình");
        btnEvaluate = new JButton("Đánh giá");
        btnShowCharts = new JButton("Hiển thị biểu đồ");
        btnSaveModel = new JButton("Lưu mô hình");
        btnLoadModel = new JButton("Load mô hình");
        btnSaveResults = new JButton("Lưu kết quả");
        btnSaveCharts = new JButton("Lưu biểu đồ");
        btnSaveAll = new JButton("Lưu tất cả");
        
        // Thiết lập trạng thái ban đầu
        btnTrain.setEnabled(false);
        btnEvaluate.setEnabled(false);
        btnShowCharts.setEnabled(false);
        btnSaveModel.setEnabled(false);
        btnLoadModel.setEnabled(true);
        btnSaveResults.setEnabled(false);
        btnSaveCharts.setEnabled(false);
        btnSaveAll.setEnabled(false);
        
        // Thêm event listeners
        btnLoadTrain.addActionListener(e -> loadTrainData());
        btnLoadTest.addActionListener(e -> loadTestData());
        btnTrain.addActionListener(e -> trainModel());
        btnEvaluate.addActionListener(e -> evaluateModel());
        btnShowCharts.addActionListener(e -> showCharts());
        btnSaveModel.addActionListener(e -> saveModel());
        btnLoadModel.addActionListener(e -> loadModel());
        btnSaveResults.addActionListener(e -> saveResults());
        btnSaveCharts.addActionListener(e -> saveCharts());
        btnSaveAll.addActionListener(e -> saveAll());
        
        // Thêm các nút vào panel theo thứ tự
        panel.add(btnLoadTrain);
        panel.add(btnLoadTest);
        panel.add(btnTrain);
        panel.add(btnEvaluate);
        panel.add(btnShowCharts);
        panel.add(btnSaveModel);
        panel.add(btnLoadModel);
        panel.add(btnSaveResults);
        panel.add(btnSaveCharts);
        panel.add(btnSaveAll);
        
        return panel;
    }
    
    /**
     * Tạo panel hiển thị kết quả
     */
    private JPanel createResultPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Kết quả"));
        
        txtResults = new JTextArea(15, 50);
        txtResults.setEditable(false);
        txtResults.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        
        JScrollPane scrollPane = new JScrollPane(txtResults);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        // Status panel
        JPanel statusPanel = new JPanel(new BorderLayout());
        lblStatus = new JLabel("Sẵn sàng - Vui lòng chọn file dữ liệu");
        lblStatus.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        statusPanel.add(lblStatus, BorderLayout.CENTER);
        
        // Progress bar
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        progressBar.setString("Sẵn sàng");
        progressBar.setVisible(false);
        
        lblProgress = new JLabel("");
        lblProgress.setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 5));
        
        JPanel progressPanel = new JPanel(new BorderLayout());
        progressPanel.add(progressBar, BorderLayout.CENTER);
        progressPanel.add(lblProgress, BorderLayout.EAST);
        progressPanel.setVisible(false);
        
        statusPanel.add(progressPanel, BorderLayout.SOUTH);
        panel.add(statusPanel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    /**
     * Tạo panel biểu đồ
     */
    private JPanel createChartPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Biểu đồ và Sơ đồ"));
        
        JLabel lblInfo = new JLabel("Nhấn 'Hiển thị biểu đồ' sau khi đánh giá mô hình để xem các biểu đồ trực quan");
        lblInfo.setHorizontalAlignment(SwingConstants.CENTER);
        lblInfo.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 14));
        lblInfo.setBorder(BorderFactory.createEmptyBorder(50, 20, 50, 20));
        
        panel.add(lblInfo, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Hiển thị các biểu đồ
     */
    private void showCharts() {
        
        if (lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng đánh giá mô hình trước khi xem biểu đồ!", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // Xóa nội dung cũ
        chartPanel.removeAll();
        
        // Tạo scroll pane cho biểu đồ
        JScrollPane scrollPane = new JScrollPane();
        JPanel chartsContainer = new JPanel();
        chartsContainer.setLayout(new BoxLayout(chartsContainer, BoxLayout.Y_AXIS));
        
        // 1. Confusion Matrix Heatmap
        JPanel confusionPanel = createConfusionMatrixChart();
        chartsContainer.add(confusionPanel);
        
        // 2. Class Accuracy Bar Chart
        JPanel accuracyPanel = createClassAccuracyChart();
        chartsContainer.add(accuracyPanel);
        
        // 3. Sample Images
        JPanel samplePanel = createSampleImagesPanel();
        chartsContainer.add(samplePanel);
        
        scrollPane.setViewportView(chartsContainer);
        chartPanel.add(scrollPane, BorderLayout.CENTER);
        
        // Chuyển sang tab biểu đồ
        tabbedPane.setSelectedIndex(1);
        
        // Refresh panel
        chartPanel.revalidate();
        chartPanel.repaint();
    }
    
    /**
     * Tạo biểu đồ Confusion Matrix
     */
    private JPanel createConfusionMatrixChart() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Confusion Matrix Heatmap"));
        
        // Kiểm tra lastEvaluation
        if (lastEvaluation == null) {
            JLabel errorLabel = new JLabel("Chưa có kết quả đánh giá để tạo confusion matrix");
            errorLabel.setHorizontalAlignment(SwingConstants.CENTER);
            panel.add(errorLabel, BorderLayout.CENTER);
            return panel;
        }
        
        double[][] confusionMatrix = null;
        try {
            confusionMatrix = lastEvaluation.confusionMatrix();
        } catch (Exception e) {
            JLabel errorLabel = new JLabel("Lỗi khi lấy confusion matrix: " + e.getMessage());
            errorLabel.setHorizontalAlignment(SwingConstants.CENTER);
            panel.add(errorLabel, BorderLayout.CENTER);
            return panel;
        }
        
        if (confusionMatrix == null || confusionMatrix.length == 0) {
            // Tạo confusion matrix giả để test
            JPanel testPanel = createTestConfusionMatrix();
            panel.add(testPanel, BorderLayout.CENTER);
            return panel;
        }
        
        int size = confusionMatrix.length;
        
        // Tạo panel cho ma trận
        JPanel matrixPanel = new JPanel(new GridLayout(size + 1, size + 1, 2, 2));
        
        // Tìm giá trị max để chuẩn hóa màu
        double maxValue = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                maxValue = Math.max(maxValue, confusionMatrix[i][j]);
            }
        }
        
        // Header row
        matrixPanel.add(new JLabel("")); // Empty corner
        for (int j = 0; j < size; j++) {
            JLabel header = new JLabel(String.valueOf(j), SwingConstants.CENTER);
            header.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
            matrixPanel.add(header);
        }
        
        // Data rows
        for (int i = 0; i < size; i++) {
            // Row label
            JLabel rowLabel = new JLabel(String.valueOf(i), SwingConstants.CENTER);
            rowLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
            matrixPanel.add(rowLabel);
            
            // Data cells
            for (int j = 0; j < size; j++) {
                JLabel cell = new JLabel(String.format("%.0f", confusionMatrix[i][j]), SwingConstants.CENTER);
                cell.setOpaque(true);
                
                // Tính màu dựa trên giá trị (từ xanh nhạt đến đỏ đậm)
                double normalizedValue = confusionMatrix[i][j] / maxValue;
                int red = (int)(255 * normalizedValue);
                int green = (int)(255 * (1 - normalizedValue));
                int blue = 100;
                
                cell.setBackground(new Color(Math.min(255, red), Math.min(255, green), blue));
                cell.setForeground(normalizedValue > 0.5 ? Color.WHITE : Color.BLACK);
                cell.setBorder(BorderFactory.createLineBorder(Color.BLACK));
                cell.setPreferredSize(new Dimension(40, 30));
                
                matrixPanel.add(cell);
            }
        }
        
        panel.add(matrixPanel, BorderLayout.CENTER);
        
        // Legend
        JPanel legendPanel = new JPanel(new FlowLayout());
        legendPanel.add(new JLabel("Legend: "));
        JLabel lowLabel = new JLabel("Low");
        lowLabel.setOpaque(true);
        lowLabel.setBackground(new Color(100, 255, 100));
        lowLabel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        lowLabel.setPreferredSize(new Dimension(30, 20));
        
        JLabel highLabel = new JLabel("High");
        highLabel.setOpaque(true);
        highLabel.setBackground(new Color(255, 100, 100));
        highLabel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        highLabel.setPreferredSize(new Dimension(30, 20));
        
        legendPanel.add(lowLabel);
        legendPanel.add(new JLabel(" → "));
        legendPanel.add(highLabel);
        
        panel.add(legendPanel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    /**
     * Tạo biểu đồ độ chính xác từng lớp
     */
    private JPanel createClassAccuracyChart() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Độ chính xác từng lớp"));
        
        if (testData == null || lastEvaluation == null) {
            JLabel errorLabel = new JLabel("Chưa có dữ liệu để vẽ biểu đồ accuracy");
            errorLabel.setHorizontalAlignment(SwingConstants.CENTER);
            panel.add(errorLabel, BorderLayout.CENTER);
            return panel;
        }
        
        int numClasses = testData.numClasses();
        JPanel chartPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                
                int width = getWidth();
                int height = getHeight();
                int barWidth = width / (numClasses + 1);
                int maxHeight = height - 60;
                
                // Vẽ trục
                g2d.setColor(Color.BLACK);
                g2d.drawLine(40, height - 40, width - 20, height - 40); // X axis
                g2d.drawLine(40, 20, 40, height - 40); // Y axis
                
                // Vẽ các cột
                for (int i = 0; i < numClasses; i++) {
                    double accuracy = lastEvaluation.precision(i);
                    int barHeight = (int)(accuracy * maxHeight);
                    int x = 50 + i * barWidth;
                    int y = height - 40 - barHeight;
                    
                    // Màu cột
                    Color barColor = new Color(100, 150, 255);
                    g2d.setColor(barColor);
                    g2d.fillRect(x, y, barWidth - 10, barHeight);
                    
                    // Viền cột
                    g2d.setColor(Color.BLACK);
                    g2d.drawRect(x, y, barWidth - 10, barHeight);
                    
                    // Nhãn lớp
                    g2d.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
                    FontMetrics fm = g2d.getFontMetrics();
                    String label = String.valueOf(i);
                    int labelWidth = fm.stringWidth(label);
                    g2d.drawString(label, x + (barWidth - 10 - labelWidth) / 2, height - 20);
                    
                    // Giá trị accuracy
                    String accText = String.format("%.3f", accuracy);
                    int accWidth = fm.stringWidth(accText);
                    g2d.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 10));
                    g2d.drawString(accText, x + (barWidth - 10 - accWidth) / 2, y - 5);
                }
                
                // Nhãn trục Y
                g2d.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
                g2d.drawString("Accuracy", 10, 30);
                
                // Nhãn trục X
                g2d.drawString("Class", width / 2 - 20, height - 5);
            }
        };
        
        chartPanel.setPreferredSize(new Dimension(600, 300));
        panel.add(chartPanel, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Tạo panel hiển thị sample images
     */
    private JPanel createSampleImagesPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Sample Images từ Test Data"));
        
        if (testData == null) {
            JLabel errorLabel = new JLabel("Chưa có dữ liệu test để hiển thị sample images");
            errorLabel.setHorizontalAlignment(SwingConstants.CENTER);
            panel.add(errorLabel, BorderLayout.CENTER);
            return panel;
        }
        
        // Kiểm tra class attribute
        if (testData.classIndex() < 0) {
            // Tạo sample images test
            JPanel testPanel = createTestSampleImages();
            panel.add(testPanel, BorderLayout.CENTER);
            return panel;
        }
        
        JPanel imagesPanel = new JPanel(new FlowLayout());
        
        // Hiển thị 10 sample images ngẫu nhiên
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
            int randomIndex = random.nextInt(testData.numInstances());
            Instance instance = testData.instance(randomIndex);
            
            JLabel imageLabel = createImageLabel(instance, randomIndex);
            imagesPanel.add(imageLabel);
        }
        
        JScrollPane scrollPane = new JScrollPane(imagesPanel);
        scrollPane.setPreferredSize(new Dimension(800, 200));
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Tạo label hiển thị image từ instance
     */
    private JLabel createImageLabel(Instance instance, int index) {
        // Tạo image 28x28 từ pixel data
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = image.createGraphics();
        
        // Vẽ pixel data (bỏ qua cột đầu tiên là label)
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int pixelIndex = y * 28 + x + 1; // +1 để bỏ qua cột label
                double pixelValue = instance.value(pixelIndex);
                int grayValue = (int)(pixelValue * 255);
                grayValue = Math.max(0, Math.min(255, grayValue)); // Clamp to 0-255
                
                Color color = new Color(grayValue, grayValue, grayValue);
                g2d.setColor(color);
                g2d.fillRect(x, y, 1, 1);
            }
        }
        
        g2d.dispose();
        
        // Scale image lên 56x56 để dễ nhìn
        Image scaledImage = image.getScaledInstance(56, 56, Image.SCALE_DEFAULT);
        JLabel imageLabel = new JLabel(new ImageIcon(scaledImage));
        
        // Thêm border và label
        imageLabel.setBorder(BorderFactory.createTitledBorder(
            "Index: " + index + ", Label: " + (int)instance.classValue()));
        
        return imageLabel;
    }
    
    /**
     * Lưu mô hình đã train
     */
    private void saveModel() {
        if (model == null || !isModelTrained) {
            JOptionPane.showMessageDialog(this, "Chưa có mô hình để lưu! Vui lòng huấn luyện mô hình trước.", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("Weka Model Files", "model"));
        fileChooser.setSelectedFile(new File("mnist_randomforest.model"));
        
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File modelFile = fileChooser.getSelectedFile();
                SerializationHelper.write(modelFile.getAbsolutePath(), model);
                
                txtResults.append("💾 Đã lưu mô hình: " + modelFile.getName() + "\n");
                txtResults.append("  - Đường dẫn: " + modelFile.getAbsolutePath() + "\n");
                txtResults.append("  - Loại mô hình: RandomForest\n\n");
                
                lblStatus.setText("Đã lưu mô hình thành công!");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi lưu mô hình: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Load mô hình đã lưu
     */
    private void loadModel() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("Weka Model Files", "model"));
        
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File modelFile = fileChooser.getSelectedFile();
                model = (Classifier) SerializationHelper.read(modelFile.getAbsolutePath());
                
                isModelTrained = true;
                btnEvaluate.setEnabled(true);
                btnPredict.setEnabled(true);
                btnSaveModel.setEnabled(true);
                
                txtResults.append("📂 Đã load mô hình: " + modelFile.getName() + "\n");
                txtResults.append("  - Đường dẫn: " + modelFile.getAbsolutePath() + "\n");
                txtResults.append("  - Loại mô hình: " + model.getClass().getSimpleName() + "\n");
                txtResults.append("  - Mô hình sẵn sàng để đánh giá và dự đoán\n\n");
                
                lblStatus.setText("Đã load mô hình thành công!");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi load mô hình: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Lưu kết quả đánh giá
     */
    private void saveResults() {
        if (lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Chưa có kết quả để lưu! Vui lòng đánh giá mô hình trước.", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("Text Files", "txt"));
        fileChooser.setSelectedFile(new File("mnist_evaluation_results.txt"));
        
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File resultsFile = fileChooser.getSelectedFile();
                PrintWriter writer = new PrintWriter(new FileWriter(resultsFile));
                
                // Ghi thông tin cơ bản
                writer.println("=== MNIST HANDWRITTEN DIGIT RECOGNITION - EVALUATION RESULTS ===");
                writer.println("Generated: " + new java.util.Date());
                writer.println();
                
                // Ghi summary
                writer.println("=== SUMMARY ===");
                writer.println(lastEvaluation.toSummaryString());
                writer.println();
                
                // Ghi confusion matrix
                writer.println("=== CONFUSION MATRIX ===");
                writer.println(lastEvaluation.toMatrixString());
                writer.println();
                
                // Ghi class details
                writer.println("=== CLASS DETAILS ===");
                writer.println(lastEvaluation.toClassDetailsString());
                writer.println();
                
                // Ghi thông tin dữ liệu
                if (testData != null) {
                    writer.println("=== DATA INFORMATION ===");
                    writer.println("Test instances: " + testData.numInstances());
                    writer.println("Attributes: " + testData.numAttributes());
                    writer.println("Classes: " + testData.numClasses());
                    writer.println("Class attribute: " + testData.classAttribute().name());
                    writer.println();
                }
                
                writer.close();
                
                txtResults.append("💾 Đã lưu kết quả đánh giá: " + resultsFile.getName() + "\n");
                txtResults.append("  - Đường dẫn: " + resultsFile.getAbsolutePath() + "\n");
                txtResults.append("  - Bao gồm: Summary, Confusion Matrix, Class Details\n\n");
                
                lblStatus.setText("Đã lưu kết quả thành công!");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi lưu kết quả: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Lưu biểu đồ thành 3 hình ảnh riêng biệt
     */
    private void saveCharts() {
        if (lastEvaluation == null || testData == null) {
            JOptionPane.showMessageDialog(this, "Chưa có dữ liệu để lưu biểu đồ! Vui lòng đánh giá mô hình trước.", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // Chọn thư mục để lưu
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        fileChooser.setDialogTitle("Chọn thư mục để lưu biểu đồ");
        
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File saveDir = fileChooser.getSelectedFile();
                String timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date());
                
                // 1. Lưu Confusion Matrix
                File confusionFile = new File(saveDir, "confusion_matrix_" + timestamp + ".png");
                JPanel confusionPanel = createConfusionMatrixChart();
                BufferedImage confusionImage = createImageFromPanel(confusionPanel);
                ImageIO.write(confusionImage, "PNG", confusionFile);
                
                // 2. Lưu Class Accuracy Chart
                File accuracyFile = new File(saveDir, "class_accuracy_" + timestamp + ".png");
                JPanel accuracyPanel = createClassAccuracyChart();
                BufferedImage accuracyImage = createImageFromPanel(accuracyPanel);
                ImageIO.write(accuracyImage, "PNG", accuracyFile);
                
                // 3. Lưu Sample Images
                File sampleFile = new File(saveDir, "sample_images_" + timestamp + ".png");
                JPanel samplePanel = createSampleImagesPanel();
                BufferedImage sampleImage = createImageFromPanel(samplePanel);
                ImageIO.write(sampleImage, "PNG", sampleFile);
                
                txtResults.append("🖼️ Đã lưu 3 biểu đồ riêng biệt:\n");
                txtResults.append("  - Thư mục: " + saveDir.getAbsolutePath() + "\n");
                txtResults.append("  1. Confusion Matrix: " + confusionFile.getName() + "\n");
                txtResults.append("  2. Class Accuracy: " + accuracyFile.getName() + "\n");
                txtResults.append("  3. Sample Images: " + sampleFile.getName() + "\n");
                txtResults.append("  - Timestamp: " + timestamp + "\n\n");
                
                lblStatus.setText("Đã lưu 3 biểu đồ thành công!");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi lưu biểu đồ: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Chuyển JPanel thành BufferedImage
     */
    private BufferedImage createImageFromPanel(JPanel panel) {
        try {
            // Tạo một JFrame ẩn để render panel
            JFrame frame = new JFrame();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(panel);
            frame.pack();
            frame.setVisible(false); // Ẩn frame
            
            // Đợi một chút để panel được render
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            // Lấy kích thước thực tế của panel
            int width = panel.getWidth();
            int height = panel.getHeight();
            
            
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g2d = image.createGraphics();
            
            // Vẽ nền trắng
            g2d.setColor(java.awt.Color.WHITE);
            g2d.fillRect(0, 0, width, height);
            
            // Vẽ panel
            panel.paint(g2d);
            g2d.dispose();
            
            frame.dispose(); // Giải phóng frame
            
            return image;
        } catch (Exception e) {
            // Tạo hình ảnh lỗi nếu có vấn đề
            BufferedImage errorImage = new BufferedImage(400, 300, BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g2d = errorImage.createGraphics();
            g2d.setColor(java.awt.Color.WHITE);
            g2d.fillRect(0, 0, 400, 300);
            g2d.setColor(java.awt.Color.RED);
            g2d.drawString("Lỗi khi tạo hình ảnh: " + e.getMessage(), 10, 150);
            g2d.dispose();
            return errorImage;
        }
    }
    
    /**
     * Tạo confusion matrix test khi có lỗi
     */
    private JPanel createTestConfusionMatrix() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Test Confusion Matrix (Demo)"));
        panel.setPreferredSize(new Dimension(500, 400));
        
        // Tạo panel đơn giản hơn
        JPanel matrixPanel = new JPanel(new GridLayout(3, 3, 5, 5));
        matrixPanel.setBackground(Color.WHITE);
        
        // Tạo 9 ô đơn giản
        Random random = new Random();
        for (int i = 0; i < 9; i++) {
            JLabel cell = new JLabel("Cell " + i, SwingConstants.CENTER);
            cell.setOpaque(true);
            
            // Màu sắc ngẫu nhiên
            Color color = new Color(random.nextInt(256), random.nextInt(256), random.nextInt(256));
            cell.setBackground(color);
            cell.setForeground(Color.BLACK);
            
            matrixPanel.add(cell);
        }
        
        panel.add(matrixPanel, BorderLayout.CENTER);
        return panel;
    }
    
    /**
     * Tạo sample images test khi có lỗi
     */
    private JPanel createTestSampleImages() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Test Sample Images (Demo)"));
        panel.setPreferredSize(new Dimension(600, 200));
        
        JPanel imagesPanel = new JPanel(new FlowLayout());
        
        // Tạo 5 hình ảnh đơn giản
        for (int i = 0; i < 5; i++) {
            JPanel imagePanel = new JPanel(new BorderLayout());
            imagePanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));
            imagePanel.setPreferredSize(new Dimension(80, 80));
            
            // Tạo label đơn giản thay vì hình ảnh
            JLabel imageLabel = new JLabel("IMG " + i, SwingConstants.CENTER);
            imageLabel.setFont(new Font("Arial", Font.BOLD, 16));
            imageLabel.setBackground(Color.LIGHT_GRAY);
            imageLabel.setOpaque(true);
            imagePanel.add(imageLabel, BorderLayout.CENTER);
            
            JLabel label = new JLabel("Label: " + i, SwingConstants.CENTER);
            imagePanel.add(label, BorderLayout.SOUTH);
            
            imagesPanel.add(imagePanel);
        }
        
        panel.add(imagesPanel, BorderLayout.CENTER);
        return panel;
    }
    
    /**
     * Lưu tất cả (model + results + charts) cùng nhau
     */
    private void saveAll() {
        if (model == null || !isModelTrained) {
            JOptionPane.showMessageDialog(this, "Chưa có mô hình để lưu! Vui lòng huấn luyện mô hình trước.", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        if (lastEvaluation == null) {
            JOptionPane.showMessageDialog(this, "Chưa có kết quả để lưu! Vui lòng đánh giá mô hình trước.", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // Chọn thư mục để lưu
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        fileChooser.setDialogTitle("Chọn thư mục để lưu tất cả");
        
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                File saveDir = fileChooser.getSelectedFile();
                String timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date());
                
                // Lưu model
                File modelFile = new File(saveDir, "mnist_model_" + timestamp + ".model");
                SerializationHelper.write(modelFile.getAbsolutePath(), model);
                
                // Lưu results
                File resultsFile = new File(saveDir, "mnist_results_" + timestamp + ".txt");
                PrintWriter writer = new PrintWriter(new FileWriter(resultsFile));
                writer.println("=== MNIST HANDWRITTEN DIGIT RECOGNITION - COMPLETE RESULTS ===");
                writer.println("Generated: " + new java.util.Date());
                writer.println();
                writer.println("=== SUMMARY ===");
                writer.println(lastEvaluation.toSummaryString());
                writer.println();
                writer.println("=== CONFUSION MATRIX ===");
                writer.println(lastEvaluation.toMatrixString());
                writer.println();
                writer.println("=== CLASS DETAILS ===");
                writer.println(lastEvaluation.toClassDetailsString());
                writer.println();
                if (testData != null) {
                    writer.println("=== DATA INFORMATION ===");
                    writer.println("Test instances: " + testData.numInstances());
                    writer.println("Attributes: " + testData.numAttributes());
                    writer.println("Classes: " + testData.numClasses());
                    writer.println("Class attribute: " + testData.classAttribute().name());
                }
                writer.close();
                
                // Lưu 3 biểu đồ riêng biệt
                File confusionFile = new File(saveDir, "confusion_matrix_" + timestamp + ".png");
                JPanel confusionPanel = createConfusionMatrixChart();
                BufferedImage confusionImage = createImageFromPanel(confusionPanel);
                ImageIO.write(confusionImage, "PNG", confusionFile);
                
                File accuracyFile = new File(saveDir, "class_accuracy_" + timestamp + ".png");
                JPanel accuracyPanel = createClassAccuracyChart();
                BufferedImage accuracyImage = createImageFromPanel(accuracyPanel);
                ImageIO.write(accuracyImage, "PNG", accuracyFile);
                
                File sampleFile = new File(saveDir, "sample_images_" + timestamp + ".png");
                JPanel samplePanel = createSampleImagesPanel();
                BufferedImage sampleImage = createImageFromPanel(samplePanel);
                ImageIO.write(sampleImage, "PNG", sampleFile);
                
                txtResults.append("💾 Đã lưu tất cả thành công!\n");
                txtResults.append("  - Thư mục: " + saveDir.getAbsolutePath() + "\n");
                txtResults.append("  - Model: " + modelFile.getName() + "\n");
                txtResults.append("  - Results: " + resultsFile.getName() + "\n");
                txtResults.append("  - Confusion Matrix: " + confusionFile.getName() + "\n");
                txtResults.append("  - Class Accuracy: " + accuracyFile.getName() + "\n");
                txtResults.append("  - Sample Images: " + sampleFile.getName() + "\n");
                txtResults.append("  - Timestamp: " + timestamp + "\n\n");
                
                lblStatus.setText("Đã lưu tất cả thành công!");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi lưu tất cả: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Tạo panel dự đoán
     */
    private JPanel createPredictPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Dự đoán chữ số"));
        
        JPanel inputPanel = new JPanel(new FlowLayout());
        inputPanel.add(new JLabel("Chọn dòng test (0-" + (testData != null ? testData.numInstances()-1 : "?") + "):"));
        
        txtTestRow = new JTextField(10);
        inputPanel.add(txtTestRow);
        
        btnPredict = new JButton("Dự đoán");
        btnPredict.setEnabled(false);
        btnPredict.addActionListener(e -> predictDigit());
        inputPanel.add(btnPredict);
        
        panel.add(inputPanel, BorderLayout.CENTER);
        
        return panel;
    }
    
    /**
     * Load dữ liệu training từ file CSV
     */
    private void loadTrainData() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("CSV Files", "csv"));
        fileChooser.setCurrentDirectory(new File("."));
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            trainFilePath = fileChooser.getSelectedFile().getAbsolutePath();
            try {
                // Load CSV data using Weka CSVLoader
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(trainFilePath));
                trainData = loader.getDataSet();
                
                // Set class attribute (first column - label)
                trainData.setClassIndex(0);
                
                // Chuyển label từ numeric sang nominal (quan trọng cho classification)
                NumericToNominal convertLabel = new NumericToNominal();
                convertLabel.setAttributeIndices("first"); // Chỉ chuyển cột đầu tiên (label)
                convertLabel.setInputFormat(trainData);
                trainData = Filter.useFilter(trainData, convertLabel);
                
                // Normalize pixel values to 0-1 range
                Normalize normalize = new Normalize();
                normalize.setInputFormat(trainData);
                trainData = Filter.useFilter(trainData, normalize);
                
                // Đảm bảo class index được set đúng sau khi filter
                trainData.setClassIndex(0);
                
                lblStatus.setText("Đã load " + trainData.numInstances() + " mẫu training");
                btnTrain.setEnabled(true);
                btnShowCharts.setEnabled(false); // Reset biểu đồ
                
                txtResults.append("✓ Đã load dữ liệu training: " + trainData.numInstances() + " mẫu\n");
                txtResults.append("  - Số thuộc tính: " + trainData.numAttributes() + "\n");
                txtResults.append("  - Class attribute: " + trainData.classAttribute().name() + " (nominal)\n");
                txtResults.append("  - Số lớp: " + trainData.numClasses() + " (0-9)\n");
                txtResults.append("  - Class values: " + trainData.classAttribute().toString() + "\n\n");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi load file training: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Load dữ liệu test từ file CSV
     */
    private void loadTestData() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("CSV Files", "csv"));
        fileChooser.setCurrentDirectory(new File("."));
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            testFilePath = fileChooser.getSelectedFile().getAbsolutePath();
            try {
                // Load CSV data using Weka CSVLoader
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(testFilePath));
                testData = loader.getDataSet();
                
                // Set class attribute (first column - label)
                testData.setClassIndex(0);
                
                // Chuyển label từ numeric sang nominal (quan trọng cho classification)
                NumericToNominal convertLabel = new NumericToNominal();
                convertLabel.setAttributeIndices("first"); // Chỉ chuyển cột đầu tiên (label)
                convertLabel.setInputFormat(testData);
                testData = Filter.useFilter(testData, convertLabel);
                
                // Normalize pixel values to 0-1 range
                Normalize normalize = new Normalize();
                normalize.setInputFormat(testData);
                testData = Filter.useFilter(testData, normalize);
                
                // Đảm bảo class index được set đúng sau khi filter
                testData.setClassIndex(0);
                
                lblStatus.setText("Đã load " + testData.numInstances() + " mẫu test");
                btnEvaluate.setEnabled(true);
                btnPredict.setEnabled(true);
                btnShowCharts.setEnabled(false); // Reset biểu đồ
                
                // Update predict panel
                updatePredictPanel();
                
                txtResults.append("✓ Đã load dữ liệu test: " + testData.numInstances() + " mẫu\n");
                txtResults.append("  - Số thuộc tính: " + testData.numAttributes() + "\n");
                txtResults.append("  - Class attribute: " + testData.classAttribute().name() + " (nominal)\n");
                txtResults.append("  - Số lớp: " + testData.numClasses() + " (0-9)\n");
                txtResults.append("  - Class values: " + testData.classAttribute().toString() + "\n\n");
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(this, "Lỗi khi load file test: " + e.getMessage(), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                e.printStackTrace();
            }
        }
    }
    
    /**
     * Huấn luyện mô hình RandomForest
     */
    private void trainModel() {
        if (trainData == null) {
            JOptionPane.showMessageDialog(this, "Vui lòng load dữ liệu training trước!", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // Chạy huấn luyện trong thread riêng để không block GUI
        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                publish("🔄 Bắt đầu huấn luyện mô hình RandomForest...\n");
                publish("📊 Thông tin dữ liệu:\n");
                publish("  - Số mẫu training: " + trainData.numInstances() + "\n");
                publish("  - Số thuộc tính: " + trainData.numAttributes() + "\n");
                publish("  - Số lớp: " + trainData.numClasses() + "\n\n");
                
                // Hiển thị progress bar
                SwingUtilities.invokeLater(() -> {
                    progressBar.setVisible(true);
                    progressBar.setIndeterminate(true);
                    progressBar.setString("Đang huấn luyện...");
                    lblProgress.setText("Đang tạo RandomForest...");
                    btnTrain.setEnabled(false);
                });
                
                publish("🔧 Tạo và cấu hình RandomForest...\n");
                Thread.sleep(500); // Delay để hiển thị
                
                // Tạo và cấu hình RandomForest
                model = new RandomForest();
                ((RandomForest)model).setNumIterations(100); // Số cây trong rừng
                ((RandomForest)model).setMaxDepth(0); // Không giới hạn độ sâu
                ((RandomForest)model).setNumFeatures(0); // Sử dụng tất cả features
                
                publish("⚙️ Cấu hình RandomForest:\n");
                publish("  - Số cây: 100\n");
                publish("  - Độ sâu tối đa: Không giới hạn\n");
                publish("  - Số features: Tất cả (" + trainData.numAttributes() + ")\n\n");
                
                SwingUtilities.invokeLater(() -> {
                    lblProgress.setText("Đang huấn luyện 100 cây...");
                });
                
                publish("🌲 Bắt đầu huấn luyện 100 cây RandomForest...\n");
                publish("⏳ Quá trình này có thể mất vài phút...\n\n");
                
                // Huấn luyện mô hình
                long startTime = System.currentTimeMillis();
                model.buildClassifier(trainData);
                long endTime = System.currentTimeMillis();
                
                publish("✅ Hoàn thành huấn luyện!\n");
                publish("⏱️ Thời gian: " + (endTime - startTime) + "ms (" + 
                       String.format("%.2f", (endTime - startTime) / 1000.0) + " giây)\n");
                publish("🎯 Mô hình sẵn sàng để dự đoán!\n\n");
                
                return null;
            }
            
            @Override
            protected void process(java.util.List<String> chunks) {
                for (String message : chunks) {
                    txtResults.append(message);
                }
                txtResults.setCaretPosition(txtResults.getDocument().getLength());
            }
            
            @Override
            protected void done() {
                try {
                    get(); // Kiểm tra lỗi nếu có
                    
                    // Cập nhật trạng thái trong EDT
                    SwingUtilities.invokeLater(() -> {
                        isModelTrained = true;
                        lblStatus.setText("Mô hình đã được huấn luyện thành công!");
                        
                        // Ẩn progress bar
                        progressBar.setVisible(false);
                        progressBar.setIndeterminate(false);
                        btnTrain.setEnabled(true);
                        btnSaveModel.setEnabled(true);
                        
                    });
                    
                } catch (Exception e) {
                    SwingUtilities.invokeLater(() -> {
                        JOptionPane.showMessageDialog(MLGuiApp.this, 
                            "Lỗi khi huấn luyện mô hình: " + e.getMessage(), 
                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                        e.printStackTrace();
                        
                        // Reset UI
                        progressBar.setVisible(false);
                        btnTrain.setEnabled(true);
                        lblStatus.setText("Lỗi khi huấn luyện mô hình");
                    });
                }
            }
        };
        
        worker.execute();
    }
    
    /**
     * Đánh giá mô hình trên tập test
     */
    private void evaluateModel() {
        
        if (!isModelTrained || testData == null || model == null) {
            String errorMsg = "Vui lòng ";
            if (!isModelTrained || model == null) {
                errorMsg += "huấn luyện mô hình ";
            }
            if (testData == null) {
                errorMsg += "load dữ liệu test ";
            }
            errorMsg += "trước!";
            
            JOptionPane.showMessageDialog(this, errorMsg, "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        // Chạy đánh giá trong thread riêng
        SwingWorker<Void, String> worker = new SwingWorker<Void, String>() {
            @Override
            protected Void doInBackground() throws Exception {
                publish("🔄 Bắt đầu đánh giá mô hình...\n");
                
                // Kiểm tra dữ liệu test
                if (testData == null) {
                    throw new Exception("Dữ liệu test chưa được load!");
                }
                
                // Đảm bảo class index được set đúng
                if (testData.classIndex() < 0) {
                    testData.setClassIndex(0);
                    publish("✓ Đã set class index = 0 (cột đầu tiên là label)\n");
                }
                
                publish("📊 Thông tin dữ liệu test:\n");
                publish("  - Số mẫu test: " + testData.numInstances() + "\n");
                publish("  - Số thuộc tính: " + testData.numAttributes() + "\n");
                publish("  - Số lớp: " + testData.numClasses() + "\n");
                publish("  - Class index: " + testData.classIndex() + "\n\n");
                
                // Hiển thị progress bar
                SwingUtilities.invokeLater(() -> {
                    progressBar.setVisible(true);
                    progressBar.setIndeterminate(true);
                    progressBar.setString("Đang đánh giá...");
                    lblProgress.setText("Đang tính toán metrics...");
                    btnEvaluate.setEnabled(false);
                });
                
                publish("⚙️ Tạo evaluation object...\n");
                Thread.sleep(300);
                
                // Tạo evaluation object với dữ liệu training (để có thông tin về classes)
                Evaluation eval = new Evaluation(trainData);
                
                publish("🔍 Đang đánh giá mô hình trên tập test...\n");
                publish("⏳ Quá trình này có thể mất vài giây...\n\n");
                
                // Đánh giá mô hình - QUAN TRỌNG: Phải gọi trước khi sử dụng các metrics
                try {
                    eval.evaluateModel(model, testData);
                    publish("✅ Đánh giá mô hình thành công!\n");
                } catch (Exception evalError) {
                    publish("❌ Lỗi khi đánh giá mô hình: " + evalError.getMessage() + "\n");
                    throw evalError; // Re-throw để SwingWorker xử lý
                }
                
                // Lưu kết quả evaluation để hiển thị biểu đồ
                lastEvaluation = eval;
                
                
                publish("✅ Hoàn thành đánh giá!\n\n");
                
                // Hiển thị kết quả chi tiết
                publish("📊 KẾT QUẢ ĐÁNH GIÁ:\n");
                publish("========================\n");
                publish("Accuracy: " + String.format("%.4f", eval.pctCorrect()) + "%\n");
                publish("Số mẫu đúng: " + (int)eval.correct() + "/" + (int)eval.numInstances() + "\n");
                publish("Số mẫu sai: " + (int)eval.incorrect() + "\n\n");
                
                // Hiển thị summary string
                publish("📋 SUMMARY:\n");
                publish("===========\n");
                publish(eval.toSummaryString() + "\n");
                
                // Hiển thị confusion matrix string
                publish("📈 CONFUSION MATRIX (STRING):\n");
                publish("=============================\n");
                publish(eval.toMatrixString() + "\n");
                
                // Hiển thị class details
                publish("📋 CLASS DETAILS:\n");
                publish("=================\n");
                publish(eval.toClassDetailsString() + "\n");
                
                // Hiển thị confusion matrix
                publish("📈 CONFUSION MATRIX:\n");
                publish("====================\n");
                try {
                    double[][] confusionMatrix = eval.confusionMatrix();
                    
                    if (confusionMatrix != null && confusionMatrix.length > 0) {
                        // Header
                        String header = "     ";
                        for (int i = 0; i < confusionMatrix.length; i++) {
                            header += String.format("%6d", i);
                        }
                        header += "\n";
                        publish(header);
                        
                        // Matrix rows
                        for (int i = 0; i < confusionMatrix.length; i++) {
                            String row = String.format("%3d: ", i);
                            for (int j = 0; j < confusionMatrix[i].length; j++) {
                                row += String.format("%6.0f", confusionMatrix[i][j]);
                            }
                            row += "\n";
                            publish(row);
                        }
                        publish("\n");
                    } else {
                        publish("Confusion matrix không khả dụng\n\n");
                    }
                } catch (Exception e) {
                    publish("Không thể tạo confusion matrix: " + e.getMessage() + "\n\n");
                }
                
                // Per-class accuracy
                publish("📋 ĐỘ CHÍNH XÁC TỪNG LỚP:\n");
                publish("========================\n");
                try {
                    for (int i = 0; i < testData.numClasses(); i++) {
                        String className = testData.classAttribute().value(i);
                        double precision = eval.precision(i);
                        double recall = eval.recall(i);
                        double fMeasure = eval.fMeasure(i);
                        
                        publish(String.format("Lớp %s: Precision=%.3f, Recall=%.3f, F1=%.3f\n", 
                                            className, precision, recall, fMeasure));
                    }
                } catch (Exception e) {
                    publish("Không thể tính toán per-class metrics: " + e.getMessage() + "\n");
                }
                publish("\n");
                
                return null;
            }
            
            @Override
            protected void process(java.util.List<String> chunks) {
                for (String message : chunks) {
                    txtResults.append(message);
                }
                txtResults.setCaretPosition(txtResults.getDocument().getLength());
            }
            
            @Override
            protected void done() {
                SwingUtilities.invokeLater(() -> {
                    try {
                        get(); // Kiểm tra lỗi nếu có
                        
                        if (lastEvaluation != null) {
                            lblStatus.setText("Đánh giá hoàn thành - Accuracy: " + String.format("%.2f", lastEvaluation.pctCorrect()) + "%");
                        } else {
                            lblStatus.setText("Đánh giá hoàn thành");
                        }
                        
                        // Ẩn progress bar và enable nút biểu đồ
                        progressBar.setVisible(false);
                        btnEvaluate.setEnabled(true);
                        btnShowCharts.setEnabled(true);
                        btnSaveResults.setEnabled(true);
                        btnSaveCharts.setEnabled(true);
                        btnSaveAll.setEnabled(true);
                        
                    } catch (Exception e) {
                        JOptionPane.showMessageDialog(MLGuiApp.this, 
                            "Lỗi khi đánh giá mô hình: " + e.getMessage(), 
                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                        e.printStackTrace();
                        
                        // Reset UI
                        progressBar.setVisible(false);
                        btnEvaluate.setEnabled(true);
                        lblStatus.setText("Lỗi khi đánh giá mô hình");
                    }
                });
            }
        };
        
        worker.execute();
    }
    
    /**
     * Dự đoán chữ số từ dữ liệu test
     */
    private void predictDigit() {
        
        if (!isModelTrained || testData == null || model == null) {
            String errorMsg = "Vui lòng ";
            if (!isModelTrained || model == null) {
                errorMsg += "huấn luyện mô hình ";
            }
            if (testData == null) {
                errorMsg += "load dữ liệu test ";
            }
            errorMsg += "trước!";
            
            JOptionPane.showMessageDialog(this, errorMsg, "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        
        try {
            String rowText = txtTestRow.getText().trim();
            if (rowText.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Vui lòng nhập số dòng test!", 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                return;
            }
            
            int rowIndex = Integer.parseInt(rowText);
            if (rowIndex < 0 || rowIndex >= testData.numInstances()) {
                JOptionPane.showMessageDialog(this, "Số dòng không hợp lệ! Phải từ 0 đến " + (testData.numInstances()-1), 
                                            "Lỗi", JOptionPane.ERROR_MESSAGE);
                return;
            }
            
            // Lấy instance từ test data
            Instance testInstance = testData.instance(rowIndex);
            
            // Dự đoán
            double prediction = model.classifyInstance(testInstance);
            double actualClass = testInstance.classValue();
            
            // Hiển thị kết quả
            txtResults.append("🔮 DỰ ĐOÁN CHỮ SỐ:\n");
            txtResults.append("==================\n");
            txtResults.append("Dòng test: " + rowIndex + "\n");
            txtResults.append("Chữ số thực tế: " + (int)actualClass + "\n");
            txtResults.append("Chữ số dự đoán: " + (int)prediction + "\n");
            txtResults.append("Kết quả: " + ((int)prediction == (int)actualClass ? "✓ ĐÚNG" : "✗ SAI") + "\n\n");
            
            lblStatus.setText("Dự đoán hoàn thành - Dòng " + rowIndex + ": " + (int)actualClass + " → " + (int)prediction);
            
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Vui lòng nhập số hợp lệ!", 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Lỗi khi dự đoán: " + e.getMessage(), 
                                        "Lỗi", JOptionPane.ERROR_MESSAGE);
            e.printStackTrace();
        }
    }
    
    /**
     * Cập nhật panel dự đoán với thông tin mới
     */
    private void updatePredictPanel() {
        if (testData != null) {
            // Cập nhật label với số dòng test có sẵn
            Component[] components = ((JPanel)((JPanel)getContentPane().getComponent(0)).getComponent(2)).getComponents();
            for (Component comp : components) {
                if (comp instanceof JPanel) {
                    Component[] subComponents = ((JPanel)comp).getComponents();
                    for (Component subComp : subComponents) {
                        if (subComp instanceof JLabel && ((JLabel)subComp).getText().contains("Chọn dòng test")) {
                            ((JLabel)subComp).setText("Chọn dòng test (0-" + (testData.numInstances()-1) + "):");
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
    
    /**
     * Main method
     */
    public static void main(String[] args) {
        // Chạy ứng dụng
        SwingUtilities.invokeLater(() -> {
            new MLGuiApp();
        });
    }
}