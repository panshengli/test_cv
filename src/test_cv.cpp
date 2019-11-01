
#include <opencv2/opencv.hpp>


#include <string>

using namespace std;

int cv_test();
int canny_operator();
int canny_video();
void colorReduce(cv::Mat image, int div);
void sharpen(const cv::Mat &image, cv::Mat &result);
void sharpen2D(const cv::Mat &image,cv::Mat &result);

// 20191029处理图像的颜色
int pic_filter();
int grayToColor();
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
    grayToColor();
    return 0;
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
int pic_filter()
{
    // 2.读取输入的图像
    cv::Mat image= cv::imread("../highway.jpeg");
    if (image.empty()) 
        return 0;
    // 3.设置输入参数
    for (int i=0;i<255;i++)
    {
        cv::floodFill(image,
                    cv::Point(100, 250),
                    cv::Scalar(i, i, 0),
                    (cv::Rect*)0,
                    cv::Scalar(35, 35, 35),
                    cv::Scalar(35, 35, 35),
                    cv::FLOODFILL_FIXED_RANGE);
        
        cv::namedWindow("result");
        cv::imshow("result",image);
        cv::waitKey(50);
    }
    cv::waitKey(0);
    return 1;
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
