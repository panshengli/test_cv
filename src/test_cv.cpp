
#include <opencv2/opencv.hpp>

#include <math.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <iostream>

using namespace std;

int cv_test();
int canny_operator();
int canny_video();
void colorReduce(cv::Mat image, int div);
void sharpen(const cv::Mat &image, cv::Mat &result);
void sharpen2D(const cv::Mat &image,cv::Mat &result);

// 处理图像的颜色20191029
cv::Mat pic_filter(cv::Mat&);
int grayToColor();
// 直方图统计像素2191101
int calc_hist();
int binary_solute();
int picFilter();
int showImage(cv::Mat );
cv::Mat getLine(cv::Mat&);
int cornerHarris();
int cornerFast();
// int surfDectector();
int cornerBrisk();
int cornerOrb();
int detectFeature2d();
int execStitcher();
int cvViz();
int getVideoFrame();
cv::Mat processCanny(cv::Mat& img, cv::Mat& out);

int main(int argc, char** argv)
{
    // cv_test();
    // canny_operator();
    // canny_video();
    // cv::Mat image,imageClone,resultSharpen,image2;
    // std::vector<cv::Mat> planes;
    // image = cv::imread("../index.jpeg");
    // // cout<< image.type()<<endl;
    // imageClone = image.clone();
    // cv::split(imageClone,planes);
    // // sharpen2D(imageClone,resultSharpen);
    // // cv::namedWindow("Sharpen2DImage");
    // cv::imshow("Sharpen2DImage",planes[0]);
    // cv::waitKey(0);
    // pic_filter();
    // grayToColor();
    // binary_solute();
    // picFilter();
    // getLine();
    // cornerHarris();
    // cornerFast();
    // surfDectector();  // 由cv_contri模块实现
    // cornerBrisk();
    // cornerOrb();
    // detectFeature2d();
    // execStitcher();
    // cvViz();
    getVideoFrame();
    return 0;
}
int getVideoFrame()
{
    // 打开视频文件
    cv::VideoCapture capture("../../../data/road_test.mp4");
    // 检查视频是否成功打开
    double rate= capture.get(CV_CAP_PROP_FPS);
    std::cout<<"fps is "<< rate<<std::endl;
    int delay = 100/rate;
    if (!capture.isOpened())
        return 1;
    // 取得帧速率
    bool stop(false);
    cv::Mat frame; // 当前视频帧
    cv::namedWindow("Extracted Frame");
    // 根据帧速率计算帧之间的等待时间,单位为 ms
    
    // 循环遍历视频中的全部帧
    clock_t s_time, e_time;
    cv::Mat result;
    while (!stop) 
    {
        // s_time=clock();    // 打印函数耗时开始
        // 读取下一帧(如果有)
        if (!capture.read(frame))
            break;
        // cv::imshow("Extracted Frame",frame); 
        // cv::imshow("Extracted Frame",processCanny(frame,result)); 
        // cv::imshow("Extracted Frame",pic_filter(frame)); 
        cv::imshow("Extracted Frame",getLine(frame)); 
        // 等待一段时间,或者通过按键停止
        if (cv::waitKey(delay)>=0)
        {
            stop = true;
        } 
        // s_time=clock();    // 打印函数耗时开始
        // sleep(1);
        // e_time=clock();    // 打印函数耗时结束
        // std::cout<<"Exec time: "<<(double)(e_time-s_time)/CLOCKS_PER_SEC*1000.<<" ms"<<std::endl;
    }
    // 关闭视频文件
    // 不是必须的,因为类的析构函数会调用
    capture.release();
    return 0;
}
cv::Mat processCanny(cv::Mat& img, cv::Mat& out)
{
    // 转换成灰度图像
    if (img.channels()==3)
    cv::cvtColor(img,out,cv::COLOR_BGR2GRAY);
    // 计算 Canny 边缘
    cv::Canny(out,out,50,200);
    // 反转图像
    cv::threshold(out,out,128,255,cv::THRESH_BINARY_INV);
    return out;
}
int cvViz()
{
    cv::Mat image1 = cv::imread("../pic_l.jpg");
    // 创建 viz 窗口
    cv::viz::Viz3d visualizer("Viz window");
    visualizer.setBackgroundColor(cv::viz::Color::white());
    // 创建一个虚拟相机
    cv::Matx33d cMatrix;
    cv::viz::WCameraPosition cam(
    cMatrix,  // 内部参数矩阵
    image1,    // 平面上显示的图像
    30.0,     // 缩放因子
    cv::viz::Color::black());
    // 在环境中添加虚拟相机
    visualizer.showWidget("Camera", cam);
    // 用长方体表示虚拟的长椅
    cv::viz::WCube plane1(cv::Point3f(0.0, 45.0, 0.0),
    cv::Point3f(242.5, 21.0, -9.0),
    true,    // 显示线条框架
    cv::viz::Color::blue());
    plane1.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    cv::viz::WCube plane2(cv::Point3f(0.0, 9.0, -9.0),
        cv::Point3f(242.5, 0.0, 44.5),
        true, // 显示线条框架
        cv::viz::Color::blue());
    plane2.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    // 把虚拟物体加入到环境中
    visualizer.showWidget("top", plane1);
    visualizer.showWidget("bottom", plane2);
    // 循环显示
    while(cv::waitKey(100)==-1 && !visualizer.wasStopped()) 
    {
        visualizer.spinOnce(1,     // 暂停 1 毫秒
        true);                     // 重绘
    }
}
int execStitcher()
{
    cv::Mat image1 = cv::imread("../pic_l.jpg");
    cv::Mat image2 = cv::imread("../pic_r.jpg");
    cv::Mat image3 = cv::imread("../1.png");
    cv::Mat image4 = cv::imread("../2.png");
    // 读取输入的图像
    std::vector<cv::Mat> images;
    images.push_back(image3);
    images.push_back(image4);
    cv::Mat panorama; // 输出的全景图
    // 创建拼接器
    cv::Stitcher stitcher = cv::Stitcher::createDefault();
    // 拼接图像
    cv::Stitcher::Status status = stitcher.stitch(images, panorama);
    showImage(panorama);
}

int detectFeature2d()
{
    cv::Mat image1 = cv::imread("../pic_l.jpg");
    cv::Mat image2 = cv::imread("../pic_r.jpg");
    cv::Mat image3 = cv::imread("../1.png");
    cv::Mat image4 = cv::imread("../2.png");
    // 定义关键点容器和描述子
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    // 定义特征检测器/描述子
    // Construct the ORB feature object
    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(60);    // 大约 60 个特征点
    // 检测并描述关键点
    // 检测 ORB 特征
    feature->detectAndCompute(image3, cv::noArray(),
    keypoints1, descriptors1);
    feature->detectAndCompute(image4, cv::noArray(),
    keypoints2, descriptors2);
    // 构建匹配器
    cv::BFMatcher matcher(cv::NORM_HAMMING); // 二值描述子一律使用 Hamming 规范
    // 匹配两幅图像的描述子
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    // 将关键点转换成 Point2f 类型
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it =
        matches.begin(); it != matches.end(); ++it)
    {
        // 获取左侧关键点的位置
        float x = keypoints1[it->queryIdx].pt.x;     //queryIdx为关键点的索引
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));
        // 获取右侧关键点的位置
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }
    // 找出 image1 和 image2 之间的本质矩阵
    cv::Mat inliers;
    cv::Mat Matrix = (cv::Mat_<double> (3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
    std::cout<<"Matrix is "<<std::endl<<Matrix<<std::endl;
    cv::Mat essential = cv::findEssentialMat(points1, points2,
        Matrix,       // 内部参数
        cv::RANSAC,   // RANSAC 方法
        0.9, 1.0, 
        inliers);     // 提取到的内点
    std::cout<<inliers.size()<<endl;
    // 根据本质矩阵还原相机的相对姿态
    cv::Mat rotation, translation;    
    cv::recoverPose(essential,      // 本质矩阵
        points1, points2,           // 匹配的关键点
        Matrix,                     // 内部矩阵
        rotation, translation,      // 计算的移动值
        inliers);                   // 内点匹配项
    std::cout<<"Rotation is "<<std::endl<<rotation<<std::endl;
    std::cout<<"Translation is "<<std::endl<<translation<<std::endl;
    // 根据旋转量 R 和平移量 T 构建投影矩阵
    cv::Mat projection2(3, 4, CV_64F); // 3×4 的投影矩阵
    rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(projection2.colRange(3, 4));
    // 构建通用投影矩阵
    cv::Mat projection1(3, 4, CV_64F, 0.); // 3×4 的投影矩阵
    cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
    diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
    // 用于存储内点
    std::vector<cv::Vec2d> inlierPts1;
    std::vector<cv::Vec2d> inlierPts2;
    // 创建输入内点的容器,用于三角剖分
    int j(0);
    for (int i = 0; i < inliers.rows; i++) 
    {
        if (inliers.at<uchar>(i)) 
        {
            inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
            inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
    }
}


    cv::Mat result_image;
    // 画出匹配线
    cv::drawMatches(image3,keypoints1,         // 第一幅图像
        image4,keypoints2,                     // 第二幅图像
        matches,
        result_image,                                      // 匹配项的向量
        cv::Scalar(255,255,255),               // 线条颜色
        cv::Scalar(255,255,255));              // 点的颜色
    // // 找到第一幅图像和第二幅图像之间的单应矩阵
    // std::vector<char> inliers;
    // cv::Mat homography= cv::findHomography(
    //     keypoints1,
    //     keypoints2, // 对应的点
    //     inliers,    // 输出的局内匹配项
    //     cv::RANSAC, // RANSAC 方法
    //     1.);        // 到重复投影点的最大距离

    // // 将第一幅图像扭曲到第二幅图像
    // cv::Mat result;
    // cv::warpPerspective(image1,        // 输入图像
    //     result,                        // 输出图像
    //     homography,                    // 单应矩阵
    //     cv::Size(2*image1.cols,image1.rows));
    // // 把第一幅图像复制到完整图像的第一个半边
    // cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
    // image2.copyTo(half);      // 把 image2 复制到 image1 的 ROI 区域
    showImage(result_image);
}
int cornerOrb()
{
    cv::Mat image = cv::imread("../highway.jpeg");
    // 关键点的向量
    std::vector<cv::KeyPoint> keypoints;
    // 构造 ORB 特征检测器对象
    cv::Ptr<cv::ORB> ptrORB =
    cv::ORB::create(75,  // 关键点的总数
        1.2,             // 图层之间的缩放因子
        8);              // 金字塔的图层数量
    // 检测关键点
    ptrORB->detect(image, keypoints);    
    // 画出关键点,包括尺度和方向信息
    cv::drawKeypoints(image,
        keypoints,                                       // 原始图像
        image,                                           // 关键点的向量
        cv::Scalar(255,255,255),                         // 结果图像
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);      // 点的颜色
    showImage(image);
}
int cornerBrisk()
{
    cv::Mat image = cv::imread("../highway.jpeg");
    // 关键点的向量
    std::vector<cv::KeyPoint> keypoints;
    // 构造 BRISK 特征检测器对象
    cv::Ptr<cv::BRISK> ptrBRISK = cv::BRISK::create();
    // 检测关键点
    ptrBRISK->detect(image, keypoints);
    // 画出关键点,包括尺度和方向信息
    cv::drawKeypoints(image,
        // 原始图像
        keypoints,
        // 关键点的向量
        image,
        // 结果图像
        cv::Scalar(255,255,255),
        // 点的颜色
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    showImage(image);
}
// // cv_contri模块
// int surfDetector()
// {
//     cv::Mat image = cv::imread("../highway.jpeg");
//     // 创建 SURF 特征检测器对象
//     cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> ptrSURF =
//     cv::xfeatures2d::SurfFeatureDetector::create(2000.0);
//     // 检测关键点
//     ptrSURF->detect(image, keypoints);
//     // 画出关键点,包括尺度和方向信息
//     cv::drawKeypoints(image,
//         // 原始图像
//         keypoints,
//         // 关键点的向量
//         featureImage,
//         // 结果图像
//         cv::Scalar(255,255,255),
//         // 点的颜色
//         cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//     showImage(image);
// }
int cornerFast()
{
    cv::Mat image = cv::imread("../highway.jpeg");
    cv::Mat gray_image;
    cv::cvtColor(image,gray_image,CV_BGR2GRAY);
    // 关键点的向量
    std::vector<cv::KeyPoint> keypoints;
    // FAST 特征检测器,阈值为 40
    cv::Ptr<cv::FastFeatureDetector> ptrFAST =
    cv::FastFeatureDetector::create(40);
    // 检测关键点
    ptrFAST->detect(image,keypoints);
    cv::drawKeypoints(image,
        keypoints,
        image,
        cv::Scalar(255,255,255),
        cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    showImage(image);
}

int cornerHarris()
{
    cv::Mat image = cv::imread("../highway.jpeg");
    cv::Mat gray_image;
    cv::cvtColor(image,gray_image,CV_BGR2GRAY);
    // 检测 Harris 角点
    cv::Mat cornerStrength;
    cv::cornerHarris(gray_image,cornerStrength,3,3,0.01);
    // 对角点强度阈值化
    cv::Mat harrisCorners;
    double threshold= 0.0001;
    cv::threshold(cornerStrength,harrisCorners,threshold,255,cv::THRESH_BINARY_INV);

    std::vector<cv::KeyPoint> keypoints;
    // GFTT检测器
    cv::Ptr<cv::GFTTDetector>ptrGFTT = 
        cv::GFTTDetector::create(
            500,
            0.01,
            10
        );
    // 检测 GFTT
    ptrGFTT->detect(image,keypoints);
    cv::drawKeypoints(image,
        keypoints,
        image,
        cv::Scalar(255,255,255),
        cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    showImage(image);
}

cv::Mat getLine(cv::Mat& image)
{
    // cv::Mat image = cv::imread("../highway.jpeg");
    // cv::resize(image,image,cv::Size(),2,2,cv::INTER_NEAREST);
    cv::Mat contours;
    cv::Canny(image,contours,150,200);
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(contours,lines,1,M_PI/180,300);

    std::vector<cv::Vec2f>::const_iterator it= lines.begin();
    while (it!=lines.end())
    {
        float rho = (*it)[0];       // 第一个元素是距离 rho
        float theta = (*it)[1];     // 第二个元素是角度 theta
        if (theta > M_PI/3. && theta < 3*M_PI/4.)
        {
            // 垂直线(大致)
            // 直线与第一行的交叉点
            cv::Point pt1(rho/cos(theta),0);     
            // 直线与最后一行的交叉点
            cv::Point pt2((rho-contours.rows*sin(theta))/cos(theta),contours.rows);  
            // 画白色的线
            cv::line( image, pt1, pt2, cv::Scalar(255), 1); 
        }
        // else
        // {
        //     // 水平线(大致)
        //     // 直线与第一列的交叉点
        //     cv::Point pt1(0,rho/sin(theta));
        //     // 直线与最后一列的交叉点
        //     cv::Point pt2(contours.cols,(rho-contours.cols*cos(theta))/sin(theta));
        //     // 画白色的线
        //     cv::line(image, pt1, pt2, cv::Scalar(150,130,2), 1);
        // }
        ++it;
    }

    return image;

}
int picFilter()
{
    cv::Mat image = cv::imread("../highway.jpeg");
    cv::Mat result,sobelX,sobelY;
    cv::Mat box_result;
    cv::Mat gauss_result;
    cv::blur(image,box_result,cv::Size(5,5));
    cv::GaussianBlur(image,gauss_result,cv::Size(7,7),2.2);//第三个参数卷积核必须是even
    cv::Mat reduced(gauss_result.rows/4,gauss_result.cols/4,CV_8U);
    for(int i=0;i<reduced.rows;i++)
    {
        for(int j=0;j<reduced.cols;j++)
        {
            reduced.at<uchar>(i,j) = image.at<uchar>(i*4,j*4);
        }
    }
    // cv::pyrUp(image,result);//上采样方法
    // cv::resize(image,result,cv::Size(),5,5,cv::INTER_NEAREST);//双线性插值
    cv::Sobel(image,sobelX,CV_8U,1,0,3,0.4,128);//设置sobel为中等灰度
    cv::medianBlur(image,result,5);
    showImage(sobelX);

}
int showImage(cv::Mat trans_image)
{
    cv::imshow("shown_win",trans_image);
    cv::waitKey(0);
    return 0;
}
int binary_solute()
{
    cv::Mat image = cv::imread("../highway.jpeg");
    cv::Mat gray_image,result;
    cv::cvtColor(image,gray_image,CV_BGR2GRAY);
    cv::Mat eroded; //erode pic default 3x3 block
    cv::Mat element(7,7,CV_8U,cv::Scalar(1,1,1));
    cv::erode(gray_image,eroded,cv::Mat(),cv::Point(1,1),3);
    cv::Mat dilated; //膨胀pics
    cv::dilate(image,dilated,cv::Mat());
    cv::morphologyEx(gray_image,result,cv::MORPH_GRADIENT,cv::Mat());
    cv::imshow("erode_win",result);
    cv::waitKey(0);
}
int calc_hist()
{
    cv::Mat image = cv::imread("../national_day.jpeg",0);
    // cv::calcHist;
}
int grayToColor()
{
    cv::Mat image = cv::imread("../national_day.jpeg",0);
    cv::Mat result;
    cv::cvtColor(image,result,CV_GRAY2BGR);
    cv::imshow("result",result);
    cv::waitKey(0);
    return 0;
}
cv::Mat pic_filter(cv::Mat& image)
{
    // // 2.读取输入的图像
    // cv::Mat image= cv::imread("../highway.jpeg");
    // if (image.empty()) 
    //     return 0;
    // // 3.设置输入参数
    for (int i=0;i<255;i++)
    {
        cv::floodFill(image,
                    cv::Point(570, 386),
                    cv::Scalar(i, i, 0),
                    (cv::Rect*)0,
                    cv::Scalar(35, 35, 35),
                    cv::Scalar(35, 35, 35),
                    cv::FLOODFILL_FIXED_RANGE);
        
        // cv::namedWindow("result");
        // cv::imshow("result",image);
        // cv::waitKey(50);
    }
    // cv::waitKey(0);
    return image;
}
void sharpen2D(const cv::Mat &image,cv::Mat &result)
{
    //构造内核(所有入口都初始化为0))
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(1,1) = 10;
    kernel.at<float>(0,1) = -1.0;
    kernel.at<float>(2,1) = -1.0;
    kernel.at<float>(1,0) = -1.0;
    kernel.at<float>(1,2) = -1.0;
    //对图像滤波
    cv::filter2D(image,result,image.depth(),kernel);
}
void sharpen(const cv::Mat &image, cv::Mat &result)
{
    result.create(image.size(),image.type());
    int nchannels = image.channels();//获取通道数
    for (int j = 1;j < image.rows-1;j++)
    {
        const uchar* previous = image.ptr<const uchar>(j-1);
        const uchar* current = image.ptr<const uchar>(j);
        const uchar* next = image.ptr<const uchar>(j+1);
        uchar* output = result.ptr<uchar>(j);   //输出行
        for (int i = nchannels;i<(image.cols-1)*nchannels;i++)
        {
            *output++ = cv::saturate_cast<uchar>(5*current[i]-current[i-nchannels]-current[i+nchannels]-previous[i]-next[i]);
        }
    }
    result.row(0).setTo(cv::Scalar(122,122,122));
    result.row(result.rows-1).setTo(cv::Scalar(122,122,122));
    result.col(0).setTo(cv::Scalar(122,122,122));
    result.col(result.cols-1).setTo(cv::Scalar(122,122,122));
}
void colorReduce(cv::Mat image, int div=64)
{
    int nl = image.rows;
    int nc = image.cols * image.channels();
    for (int j=0; j<nl; j++)
    {
        uchar * data = image.ptr<uchar>(j);
        for (int i=0;i<nc; i++ )
        {   
            // 减色计算利用整数除法特性
            data[i] = data[i]/div*div +div/2;
        }
    }
}
int cv_test()
{
    cv::Mat M1(3000,2000, CV_8UC1, cv::Scalar(100));
    M1.row(2000) = M1.row(1200)*2;
    cout << M1.row(20) << endl;
    uchar value1 = M1.at<uchar>(150,100);
    M1.at<uchar>(150,100)=0; //将第 150 行第 100 列像素值设置为 0
    cv::imshow("creat_img",M1);
    cv::waitKey(0);

    cv::Mat M3(300,200, CV_8UC3, cv::Scalar(0,0,255));
    // cout << M3 << endl;
    uchar value3 = M3.at<uchar>(150,100);
    M3.at<uchar>(150,100)=0; //将第 150 行第 100 列像素值设置为 0
    cv::imshow("creat_img3",M3);
    cv::waitKey(0);
    return 0;
}
int canny_operator()
{
    cv::Mat im = cv::imread("../index.jpeg", 0);
    if( im.empty() )
    {
        cout << "Can not load image." << endl;
        return -1;
    }
    cv::Mat result;
    cv::Canny(im, result, 50, 255);
    //保存结果
    cv::imwrite("kobe-canny.png", result);
    cv::imshow("index_img",result);
    cv::waitKey(0);
    return 0;
}
int canny_video()
{
    using namespace cv;
    //打开第一个摄像头
    //VideoCapture cap(0);
    //打开视频文件
    VideoCapture cap(0);
    //检查是否成功打开
    if(!cap.isOpened())
    {
        cerr << "Can not open a camera or file." << endl;
        return -1;
    }
    Mat edges;
    //创建窗口
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        //从 cap 中读一帧,存到 frame
        cap >> frame;
        //如果未读到图像
        if(frame.empty())
            break;
        //将读到的图像转为灰度图
        cvtColor(frame, edges, CV_BGR2GRAY);
        //进行边缘提取操作
        Canny(edges, edges, 0, 30, 3);
        //显示结果
        imshow("edges", edges);
        //等待 30 秒,如果按键则推出循环
        if(waitKey(30) >= 0)
        break;
    }
    //退出时会自动释放 cap 中占用资源
    return 0;
}
