//
//  Bachelor's Thesis
//
//  UNIVERSITY OF VALENCIA
//
//  Title: Developement of an application for the classification of butterfly wings using digital images.
//
//  Student: Álvaro Martínez Fernández
//
//  Bachelor's thesis coordinator: Esther Durá Martínez
//
//  ----------------------------
//
//  This is the code used for the developement of this project
//
//  main.cpp
//  OpenCV
//
//  Created by Álvaro Martínez Fernández on 11/2/16.
//


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

/**
 * VARIABLES GLOBALES
 **/
Mat src;
Mat src_hsv;
RNG rng(12345);
string url_read, url_write, nombre;
// Vector de vectores de características para cada una de las muestras
vector<vector<float> > features;
vector<float > features_aux;

// Si está en true, se crea la matriz con todos los vectores de características, si no se alamcenan en cuanto a las
// características de textura los métodos de Gabor y de granulometría los valores estadísticos y de función
bool HAS_ALL_TEXTURE_METHODS = true;

/**
 * DECLARACIÓN DE FUNCIONES
 **/
Mat ScaleImage( const Mat&, int );
Mat BackgroundExtraction ( const Mat&, Mat&, Scalar, Scalar );
int LineDetection( const Mat& );
Mat ImageSegmentation(int, const Mat&, Mat& );
float ShapeFeauturesExtraction(const Mat&, int, float, float);
void BorderFeaturesExtraction ( const Mat&, int, float );
void ColorFeaturesExtraction( const Mat& );
void CreateFeatureMatrix();
void Classifier();

/**
 * IMÁGENES DE MUESTRA - Deben coincidir con el nombre de la imagen del direcorio donde se encuentran las imágenes de muestra
 **/
string image[] =   {"aegeria_1", "aegeria_2", "aegeria_3", "aegeria_4", "dalmatana", "dalmatana0001", "dalmatana0002", "dalmatana0003", "dalmatana0004", "dalmatana0005", "dubia1", "dubia2", "dubia3", "dubia4", "lachesis_1", "lachesis_2", "lachesis_3", "lachesis_4", "pomonella_1", "pomonella_0002", "pomonella_0004", "pomonella_0005", "tigrina1", "tigrina2", "tigrina3", "tigrina4"};

/**
 * PROGRAMA PRINCIPAL
 * En esta función se procesan cada una de las imágenes y finalmente se clasifican con k-means, mostrando
 * los resultados de la clasificación por consola.
 **/
int main( int argc, char** argv )
{
    for (int i = 0; i < 26; i++) { //26 MUESTRAS EN TOTAL
        string folder = image[i];
        string image_name = image[i];
        nombre = image_name;
        
        cout << "Procesando: \t" << image_name << endl;
        
        /**
         * DIRECTORIO DONDE SE ENCUENTRAN LAS MUESTRAS
         **/
        url_read = "/Users/kolibrid111/Documents/Universidad/Cuarto/TFG//MARIPOSAS/wings_1/";
        
        /**
         * DIRECTORIO DONDE SE ALMACENAN LOS RESULTADOS
         **/
        url_write = "/Users/kolibrid111/Documents/Universidad/Cuarto/TFG/Resultados/";
    
        folder.erase(remove_if(folder.begin(), folder.end(), [](char c) { return !isalpha(c); } ), folder.end());
        folder[0] = toupper(folder[0]);
        
        url_read = url_read + image_name + ".png";
        url_write  = url_write + folder + "/" + image_name + "/";

        src = imread(url_read, 1);
        
        src = ScaleImage(src, 700);
        
        imwrite( url_write + "0_source.png", src );
    
        // ****************************************************
        // * 1. EXTRACCIÓN DEL FONDO
        // ****************************************************
        
        Mat bg_mask;
        Mat src_contours;
        
        src_contours = BackgroundExtraction(src, bg_mask, Scalar(0, 0, 100), Scalar(140, 50, 205));
        
        // Aquí utilizamos otros valores para extraer el fondo, ya que las muestras (10 - 17) tienen un color de fondo
        // distinro al resto
        if(i > 9 && i < 18){
            src_contours = BackgroundExtraction(src, bg_mask, Scalar(0, 0, 0), Scalar(140, 17, 200));
        }
        
        // Aplicamos la máscara a la imagen principal
        Mat src_bg;
        src.copyTo(src_bg, bg_mask);
        
        // ****************************************************************
        // * 2. SEGMENTACIÓN DE LAS ALAS
        // ****************************************************************
        // Primero buscamos el punto donde se separan las alas del torso de la mariposa utilizando HoughLines
        int x_segment;
        x_segment = LineDetection(src_contours);
        
        // A continuación segmentamos el ala
        Mat segmented_mask;
        segmented_mask = ImageSegmentation(x_segment, bg_mask, src_bg);
        
        // Aplicamos la mascara
        Mat src_segmented;
        src_bg.copyTo(src_segmented, segmented_mask);
        
        // Alamacenamos la imagen en el directorio de RESULTADOS
        imwrite( url_write + "3_5_src_segmented.png", src_segmented );
        
        // ****************************************************
        // * 3. EXTRACCIÓN DE CARACTERÍSTICAS DE FORMA
        // ****************************************************
        Mat src_shapes;
        
        // Convertimos la imagen a escala de grises
        cvtColor( src_segmented, src_shapes, CV_BGR2GRAY );
        imwrite( url_write + "4_1_src_gray.png", src_shapes );
        
        // Desenfocamos la imagen
        blur( src_shapes, src_shapes, Size(3,3) );
        imwrite( url_write + "4_2_src_blured.png", src_shapes );
        
        // Obtenemos el valor de la distancia entre las formas porque hay que dividirlo en el siguiente apartado
        // entre la longitud del ala
        float distance = ShapeFeauturesExtraction(src_shapes, 40, 15.0, 150.0);
        
        // ****************************************************
        // * 4. EXTRACCIÓN DE CARACTERÍSTICAS DEL CONTORNO DEL ALA
        // ****************************************************
        // Está función geera un fichero con los valores del contorno para poder procesarolos con la transformada
        // de Fourier en Matlab
        BorderFeaturesExtraction(segmented_mask, x_segment, distance);
        
        // ****************************************************
        // * 5. EXTRACCIÓN DE CARACTERÍSTICAS DE COLOR
        // ****************************************************
        ColorFeaturesExtraction(segmented_mask);
        
        // ****************************************************
        // * 6. EXTRACCIÓN DE CARACTERÍSTICAS DE TEXTURA
        // ****************************************************
        // Utilizamos Final mask y src_gray en Matlab para procesarlas con los métodos de Juan Domingo
        
        // Almacenamos el vator de características y lo limpiamos para procesar la siguiente muestra
        features.push_back(features_aux);
        features_aux.clear();
    }
    
    // ****************************************************
    // * 7. CLASIFICACIÓN DE LAS ALAS
    // ****************************************************
    // Creamos la matriz de características
    CreateFeatureMatrix();
    
    // Clasificamos las imágenes
    Classifier();
    
    //waitKey(0);
    
    return(0);
}


/**
  * ScaleImage - Función para escalar una imagen.
  * Le pasamos la imagen el ancho que queremos que tenga.
 */
Mat ScaleImage( const Mat& img, int target_width = 500 )
{
    int width = target_width;
    
    float scale = ( ( float ) target_width ) / img.cols;
    
    int height = img.rows * scale;
    
    // Cuando creamos un imagen Mat, definimos primero las filas (row), y a continuación las columnas (cols)
    // es decir, primero el alto y luego el ancho
    Mat square = Mat::zeros( height, width, img.type() );
    
    Rect roi;
    
    roi.width = width;
    roi.x = 0;
    roi.height = height;
    roi.y = 0;
    
    resize( img, square( roi ), roi.size() );
    
    return square;
}

/**
 * BackgroundExtraction - Extracción del fondo de la imagen
 */
Mat BackgroundExtraction ( const Mat& src, Mat& bg_mask, Scalar lowerb, Scalar upperb ){
    Mat src_hsv;
    
    // Convertimos la imagen al espacio de color HSV, para trabajar mejor
    // en la selección del color de fondo
    cvtColor(src,src_hsv,CV_BGR2HSV);
    
    // *******************************************************
    // * Selección de píxeles grisaceos mediante inRange
    // *******************************************************
    Mat color_mask;
    
    // Obtenemos la máscara con la selección del fondo gris
    inRange(src_hsv, lowerb, upperb, color_mask);
    imwrite( url_write + "1_1_inRange_mask.png", color_mask );
    
    // *******************************************************
    // * Limpieza de la máscara mediante filtros morfológicos
    // *******************************************************
    Mat open_mask, close_mask;
    
    // Generamos un elemento estructurante
    Mat structElement = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1,1));
    
    // Aplicamos un filtro morfológico de apertura
    morphologyEx( color_mask, open_mask, MORPH_OPEN, structElement );
    imwrite( url_write + "1_2_open_mask.png", open_mask );
    
    // Aplicamos un filtro morfológico de cierre
    morphologyEx( open_mask, close_mask, MORPH_CLOSE, structElement );
    imwrite( url_write + "1_3_close_mask.png", close_mask );
    
    // *******************************************************
    // * Selección del area de la mariposa
    // *******************************************************
    Mat1i labels;    Mat1i stats;   Mat1d centroids;
    Mat biggestAreaMask;
    
    // Cogemos el área del exterior de la mariposa
    //compare(labels, 0, biggestAreaMask, CMP_EQ);
    biggestAreaMask = 255 - close_mask;
    
    imwrite( url_write + "1_4_biggestArea_bg.png", biggestAreaMask );
    
    
    // *******************************************************
    // * Dibujamos los contornos de la mariposa
    // *******************************************************
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int idx = 0;
    
    // Creamos una imagen con el mismo tamaño que todas las imágenes con las que se trabaja
    Mat dst_contours = Mat::zeros(biggestAreaMask.rows, biggestAreaMask.cols, CV_8UC3);
    
    // Generamos los contornos
    findContours( biggestAreaMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    
    // Dibujamos los contornos
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( 0, 255, 0 );
        drawContours( dst_contours, contours, idx, color, 1, 8, hierarchy );
    }
    
    imwrite( url_write + "1_5_contours_bg.png", dst_contours );
    
    // *************************************************************************
    // * Rellenado de los contornos para que no queden píxeles sin seleccionar
    // *************************************************************************
    fillPoly(biggestAreaMask, contours, Scalar(255, 255, 255));
    
    imwrite( url_write + "1_6_filledContours_bg.png", biggestAreaMask );
    
    biggestAreaMask.copyTo(bg_mask);
    
    return dst_contours;
}

/**
 * LineDetection - Algoritmo que busca el punto donde
 * se separa la cabeza de las alas de la mariposa
 */
int LineDetection( const Mat& src_contours ){
    Mat src_lines, color_lines;
    
    // Aplicamos el filtro de Canny
    Canny( src_contours, src_lines, 120, 250, 3 );
    
    // Convertimos la imagen al espacio RGB
    cvtColor( src_lines, color_lines, CV_GRAY2BGR );
    
    imwrite( url_write + "2_1_canny.png", src_lines );
    
    vector<Vec4i> lines;
    Vec4i body_line, body_line_aux;
    Point v, v_aux;
    float module, module_aux;
    float ang_aux;
    bool first = true;
    RNG rng(12345);
    
    // Aplicamos la transformada de Hough
    HoughLinesP( src_lines, lines, 2, 2*CV_PI/160, 50, 80, 30 );
    
    // Recorremos cada una de las líneas detectadas y buscamos la línea que se encuentra en el borde superior del ala
    // de la mariposa, más cercana al torso
    for( size_t i = 0; i < lines.size(); i++ )
    {
        ang_aux = atan( ((float)lines[i][3] - lines[i][1])/((float)lines[i][2] - lines[i][0]) )*180/CV_PI;
        // La línea que queremos encontrar tiene un ángulo mayor que -60 grados y menos que 0 grados
        if(ang_aux > -60 && ang_aux < 0){
            if(first){
                body_line = lines[i];
                first = false;
                line( color_lines, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 0, 0), 3, 8 );
            }else{
                line( color_lines, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 0, 0), 3, 8 );
                body_line_aux = lines[i];
                
                v.y = body_line[1] - body_line[3];
                v.x = body_line[0] - body_line[2];
                module = sqrt( v.x*v.x + v.y*v.y );
                
                v_aux.y = body_line_aux[1] - body_line_aux[3];
                v_aux.x = body_line_aux[0] - body_line_aux[2];
                module_aux = sqrt( v_aux.x*v_aux.x + v_aux.y*v_aux.y );
                
                // Seleccionamos la línea que esté más cerca de la esquina superior izquierda, con un margen de 65px
                if (body_line_aux[0] < body_line[0] && body_line_aux[0] > 65){
                    if(module_aux > module || body_line_aux[1] > body_line[1]){
                        body_line = body_line_aux;
                    }
                    
                }
            }
        }
        
        // Dibujar todas las líneas
        //line( color_lines, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8 );
    }
    
    // Pintamos la línea que buscámbamos de color verde
    line( color_lines, Point(body_line[0], body_line[1]), Point(body_line[2], body_line[3]), Scalar(0, 255, 0), 3, 8 );
    
    // Aprovechamos aquí para calcular la pendiente del ala y la guardamos en el vector
    float pendiente = ( (float)(src.rows - body_line[1]) - (src.rows - body_line[3]) ) / ((float)body_line[0] - body_line[2]);

    features_aux.push_back(pendiente);
    
    imwrite( url_write + "2_2_lines.png", color_lines );
    
    return body_line[0];
}

/**
 * ImageSegmentation - Segmentación de la imagen. Se aislan las alas
 */
Mat ImageSegmentation(int x, const Mat& bg_mask, Mat& src_bg ){
    
    Mat mask_segmented(src.size(), CV_8UC1, Scalar::all(0));
    
    // *******************************************************
    // * Generación de un rectángulo que correponde a la parte
    // * de la imagen que se quiere ocultar
    // *******************************************************
    Rect ROI_segmented( x, 0, src.cols - x, src.rows );
    
    // Añadimos el rectánguo a la máscara creada
    mask_segmented(ROI_segmented).setTo(Scalar::all(255));
    imwrite( url_write + "3_1_rectangle_mask.png", mask_segmented );
    
    // La imagen src queda de la siguiente forma con la nueva máscara
    Mat src_segmented;
    src_bg.copyTo(src_segmented, mask_segmented);
    imwrite( url_write + "3_2_src_rectangle.png", src_segmented );
    
    // *******************************************************
    // * Combinación de todas las máscaras procesadas
    // *******************************************************
    // Creamos un nuevo rectángulo para poder aplicarlo a l máscara
    // del fondo
    Rect ROI_segmented2( 0, 0, x, src_bg.rows );
    // Aplicamos el rectángulo a la máscara que habíamos obtenido
    bg_mask(ROI_segmented2).setTo(Scalar::all(0));
    imwrite( url_write + "3_3_mask_fusion.png", bg_mask );
    
    // *******************************************************
    // * Limpieza de la máscara
    // *******************************************************
    Mat1i labels2;
    Mat1i stats2;
    Mat1d centroids2;
    
    // Generación de estadísticas de los componentes conectados
    connectedComponentsWithStats(bg_mask, labels2, stats2, centroids2);
    
    // El siguiente algoritmo selecciona el áre más grande de a máscara
    // El resto de áreas suelen ser elementos que se han quedado sueltos
    // al segmentar la imagen. Tales como antenas.
    int label = 0;
    long biggestArea = 0;
    long area;
    int widthComponent;
    for (int i = 0; i < stats2.rows; i++){
        area = stats2.at<__int32_t>(i, CC_STAT_AREA);
        widthComponent = stats2.at<__int32_t>(i, CC_STAT_WIDTH);
        if(area > biggestArea && widthComponent < src.cols){
            biggestArea = area;
            label = i;
        }
    }
    
    // Guardamos el área más grande en una imagen
    Mat biggestAreaMask;
    compare(labels2, label, biggestAreaMask, CMP_EQ);
    
    imwrite( url_write + "3_4_final_mask.png", biggestAreaMask );
    
    return biggestAreaMask;
}

/**
 * ShapeFeauturesExtraction - Se extraen características de forma del ala
 */
float ShapeFeauturesExtraction( const Mat& src_shapes, int thresh, float min_radius, float max_radius ){
    int num_shapes = 0;
    
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point2f> formas;
    
    /// Detect edges using Threshold
    threshold( src_shapes, threshold_output, thresh, 255, THRESH_BINARY );
    imwrite( url_write + "4_3_threshold.png", threshold_output );
    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    
    
    /// Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        if((int)radius[i] > min_radius && (int)radius[i] < max_radius){
            circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
            num_shapes++;
            formas.push_back(center[i]);
        }
    }
    
    /**
     * Calculamos las características de forma
     **/
    Point vector;
    float distance = 0, mod;
    
    // En caso de que hayan dos formas, calculamos la distancia entre ellas.
    // Si no calculamos la suma de las distancias entre ellas.
    // Una vez tenemos la suma de las distancias, dividimos esta suma entre la
    // longitud del ala. El valor de la longitud del ala se calcula en la función
    // BorderFeatureExtraction.
    if(num_shapes == 2){
        vector.x = formas[1].x - formas[0].x;
        vector.y = formas[1].y - formas[0].y;
        mod = sqrt(vector.x*vector.x + vector.y*vector.y);
        distance += mod;
    }else if (num_shapes > 2){
        
        for(int i = 0; i < formas.size() - 1; i++){
            for (int j = i+1; j < formas.size(); j++){
                vector.x = formas[j].x - formas[i].x;
                vector.y = formas[j].y - formas[i].y;
                mod = sqrt(vector.x*vector.x + vector.y*vector.y);
                distance += mod;
            }
        }
        
    }
    
    imwrite( url_write + "4_4_shapes.png", drawing );
    
    features_aux.push_back(num_shapes);
    
    return distance;
}

/**
 * BorderFeaturesExtraction - Se extraen características del contorno del ala
 * En esta función se extraen los contornos del ala, y se crea un fichero con los valores
 * de la posición de cada punto del contorno y el ángulo
 */
void BorderFeaturesExtraction ( const Mat& segmented_mask, int x, float distance ){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int idx = 0;
    ofstream file;
    
    // Creamos una imagen con el mismo tamaño que todas las imágenes con las que se trabaja
    Mat dst_contours = Mat::zeros(src.rows, src.cols, CV_8UC3);
    
    // Generamos los contornos
    findContours( segmented_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    
    // Dibujamos los contornos
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( 0, 255, 0 );
        drawContours( dst_contours, contours, idx, color, 1, 8, hierarchy );
    }
    
    imwrite( url_write + "7_1_final_contours.png", dst_contours );
    
    // Recorremos los contornos
    Point p0, p1, v;
    float mod, m, ang;
    int ini = 0, fin = 0;
    int height = segmented_mask.rows;
    int aux;
    
    p0.x = x;
    p0.y = 2;
    
    // Cálculo de la longitud del ala
    Point longitud1, longitud2, vecLong;
    float longAla;
    
    longitud1 = contours[0][0];
    
    for(int j = 0; j < contours[0].size(); j++){
        p1 = contours[0][j];
        p1.y = height - p1.y;
        aux = height - p1.y;
        //Vector
        v.x = p1.x - p0.x;
        v.y = p1.y - p0.y;
        mod = sqrt( v.x*v.x + v.y*v.y );
        m = (float)(p1.y - p0.y)/(p1.x - p0.x);
        ang = atan(m)*180/CV_PI ;
        if(ang != ang) ang = 0;
        if(ang == 0)
        {
            ini = j;
            p1 = contours[0][ini];
        }
        
        if(isinf(-m) && fin == 0){
            fin = j + 1;
            longitud2 = contours[0][j];
        }
    }
    
    // Longitud del ala
    vecLong.x = longitud2.x - longitud1.x;
    vecLong.y = longitud2.y - longitud1.y;
    longAla = sqrt(vecLong.x*vecLong.x + vecLong.y*vecLong.y);
    
    // Abrimos el fichero
    file.open (url_write + nombre + ".m");
    file << "M=[\n";
    for(int j = ini; j < contours[0].size(); j++){
        p1 = contours[0][j];
        p1.y = height - p1.y;
        
        //Vector
        v.x = p1.x - p0.x;
        v.y = p1.y - p0.y;
        mod = sqrt( v.x*v.x + v.y*v.y );
        m = (float)(p1.y - p0.y)/(p1.x - p0.x);
        ang = atan(m);
        
        if(ang != ang) ang = 0;
        
        file << 2*ang << " " << mod << "\n";
    }
    
    for(int j = 0; j < fin; j++){
        p1 = contours[0][j];
        p1.y = height - p1.y;
        
        //Vector
        v.x = p1.x - p0.x;
        v.y = p1.y - p0.y;
        mod = sqrt( v.x*v.x + v.y*v.y );
        m = (float)(p1.y - p0.y)/(p1.x - p0.x);
        ang = atan(m);
        
        if(ang != ang) ang = 0;
        
        if(2*ang == 3.14159)
            file << 2*ang << " " << mod << "\n";
        
    }
    
    file << "];";
    
    features_aux.push_back( (float)distance/longAla );
    
    file.close();
}

/**
 * ColorFeaturesExtraction - Extracción de características de color
 */
void ColorFeaturesExtraction( const Mat& segmented_mask ){
    Mat hsv_hist;
    
    cvtColor(src, hsv_hist, CV_BGR2HSV);
    
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 70, sbins = 70;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};
    
    calcHist( &hsv_hist, 1, channels, segmented_mask,
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    
    int scale = 10;
    Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
    int aIntensity[hbins][sbins];
    
    for( int h = 0; h < hbins; h++ )
        for( int s = 0; s < sbins; s++ )
        {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            aIntensity[h][s] = intensity;
            rectangle( histImg, Point(h*scale, s*scale),
                      Point( (h+1)*scale - 1, (s+1)*scale - 1),
                      Scalar::all(intensity),
                      CV_FILLED );
        }
    
    
    // Cálculo de características
    int min_intensity = 25;
    int n_group = 0;
    Mat groupsImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
    bool vHue[hbins];
    int vGroup[hbins];
    bool isHue = false;
    
    for( int h = 0; h < hbins; h++ )
    {
        vHue[h] = false;
        for( int s = 0; s < sbins; s++ )
        {
            if(aIntensity[h][s] > min_intensity){
                vHue[h] = true;
                s = sbins;
            }
        }
    }
    
    // Merge de los grupos
    for( int h = 0; h < hbins; h++ )
    {
        if(vHue[h] == false){
            if(vHue[h-1] == true && vHue[h+1] == true){
                vHue[h] = true;
            }
        }
    }
    
    
    // Asignamos los grupos
    for( int h = 0; h < hbins; h++ )
    {
        if(vHue[h] == true){
            if(!isHue){
                n_group++;
                isHue = true;
            }
            vGroup[h] = n_group;
        }else{
            if(isHue){
                isHue = false;
            }
            vGroup[h] = 0;
        }
    }
     
    // Empezamos a calcular algunos valores para la media de la distribución,
    // la vrianza y contamos también el número valores de cada cluster.
    // Los valores de la media de la distribución son sumH y sumS.
    // Los valores para la varianza son varH y varS
    // En sum se cuentan los valores de cada cluster, es decir los cuadraditos que hay
    // porque queremos saber cual será el cluster mayor.
    // Una solución que propongo es encontrar el cluster mayor y luego calcular la media
    // de la distribución y la varianza. Pero en este caso la mayoria de los histogramas solo tienen
    // un cluster por lo que no afecta mucho al rendimiento.
    int sumH[n_group], sumS[n_group], sum[n_group], varH[n_group], varS[n_group];
    int N[n_group];
    vector<Point> distribution(n_group);
    
    for (int i = 0; i < n_group; i++) {
        sumH[i] = sumS[i] = sum[i] = N[i] = varH[i] = varS[i] = 0;
    }
    
    for( int h = 0; h < hbins; h++ )
    {
        for( int s = 0; s < sbins; s++ )
        {
            if(aIntensity[h][s] > min_intensity){
                rectangle( groupsImg, Point(h*scale, s*scale),
                          Point( (h+1)*scale - 1, (s+1)*scale - 1),
                          Scalar::all(255),
                          CV_FILLED );
                putText(groupsImg, to_string(vGroup[h]),
                        Point(h*scale, s*scale),
                        FONT_HERSHEY_SIMPLEX, 1,
                        Scalar(0, 0, 255));
                
                float binVal = hist.at<float>(h, s);
                int intensity = cvRound(binVal*255/maxVal);
                
                N[ vGroup[h] - 1 ] += intensity;
                
                sumH[ vGroup[h] - 1 ] += (float)(h * intensity);
                sumS[ vGroup[h] - 1 ] += (float)(s * intensity);
                
                sum[ vGroup[h] - 1 ] += 1;
            }
        }
    }
    
    
    
    // Buscamos el mayor cluster de color
    int biggest_idx = 0, max_num = 0;
    
    for (int i = 0; i < n_group; i++) {
        if(sum[i] > max_num){
            max_num = sum[i];
            biggest_idx = i;
        }
    }
    
    // Calculamos la media de la distribución del mayor cluster
    Point pto;
        
    pto.x = sumH[biggest_idx] / N[biggest_idx];
    pto.y = sumS[biggest_idx] / N[biggest_idx];
        
    distribution.at(biggest_idx) = pto;
    
    for( int h = 0; h < hbins; h++ )
    {
        for( int s = 0; s < sbins; s++ )
        {
            if(aIntensity[h][s] > min_intensity){
                float binVal = hist.at<float>(h, s);
                int intensity = cvRound(binVal*255/maxVal);
                
                int idx = vGroup[h] - 1;
                
                varH[idx] += ( (float)(h * h * intensity)/N[idx] - pto.x*pto.x );
                varS[idx] += ( (float)(s * s * intensity)/N[idx] - pto.y*pto.y );
            }
        }
    }
    
    // Calculamos la varianza sopesada
    Point variance;
    int sumVarH = 0, sumVarS = 0;
    
    for(int i = 0; i < max_num; i++){
        sumVarH += pto.x;
        sumVarS += pto.y;
    }
    
    variance.x = varH[biggest_idx];//varH[biggest_idx] - sumVarH;
    variance.y = varS[biggest_idx];//varS[biggest_idx] - sumVarS;
    
    imwrite( url_write + "5_1_h-s-histogram.png", histImg );
    
    imwrite( url_write + "5_2_group_histogram.png", groupsImg );
    
    // Añadimos los valores al vector de características
    features_aux.push_back(n_group);
    features_aux.push_back(pto.x);
    features_aux.push_back(pto.y);
    features_aux.push_back(variance.x);
    features_aux.push_back(variance.y);
}

/**
 * CreateFeatureMatrix - Construcción de la matriz de características
 */
void CreateFeatureMatrix(){
    string matrixFile = "featureMatrix.txt";
    
    if(!HAS_ALL_TEXTURE_METHODS)
        matrixFile = "featureMatrixWithoutTexMethods.txt";
    
    ofstream featureMatrix;
    
    // En este directorio se almacena la matriz de características
    featureMatrix.open("/Users/kolibrid111/Documents/Universidad/Cuarto/TFG/Matrix/"  + matrixFile);
    
    ifstream textureFeatures("/Users/kolibrid111/Documents/Universidad/Cuarto/TFG/Matrix/texture");
    ifstream borderFeatures("/Users/kolibrid111/Documents/Universidad/Cuarto/TFG/Matrix/border");
    
    string line;
    int id;
    int cont;
    
    // Almacenamos las características de textura
    while(getline(textureFeatures, line))
    {
        vector<float>   lineData;
        stringstream  lineStream(line);
        float value;
        cont = 1;
        
        lineStream >> id;
        
        while(lineStream >> value)
        {
            if(HAS_ALL_TEXTURE_METHODS)
                features[id].push_back(value);
            else
                // Seleccionamos los métodos de gabo y granulometría valores esradísticos y de función
                if( (cont >=1 && cont <= 12) || (cont >= 23 && cont <= 44) )
                    features[id].push_back(value);
            
            cont ++;
        }
    }
    
    // Almacenamos características de borde
    while(getline(borderFeatures, line))
    {
        vector<float>   lineData;
        stringstream  lineStream(line);
        float value;
        
        lineStream >> id;
        
        while(lineStream >> value)
        {
            features[id].push_back(value);
        }
        
    }
    
    // Almacenamos las características de pendiente, forma y color
    for(int i = 0; i < features.size(); i++){
        for(int j = 0; j < features[i].size(); j++){
            featureMatrix << features[i][j] << " ";
        }
        
        if(i < features.size() - 1)
            featureMatrix << "\n";
     }
    
    featureMatrix.close();
    textureFeatures.close();
}


/**
 * Classifier - Se clasifican las mariposas por características
 */
void Classifier(){
    string matrixFile = "featureMatrix.txt";
    Mat data(26, 91, CV_32F);
    int clusterCount = 3;
    Mat labels;
    Mat centers;
    Mat points(5, 1, CV_32FC2);
    Mat img(500, 500, CV_8UC3);
    
    string line;
    int i = 0;
    
    // En este directorio se alamcenarán los resultados. Se deben crear las carpetas para cada uno de los experimentos
    // y clusters
    string experimentationPath = "/Users/kolibrid111/Documents/Universidad/Cuarto/TFG/Experimentation/";
    experimentationPath  = experimentationPath + "cluster" + to_string(clusterCount);
    
    if(!HAS_ALL_TEXTURE_METHODS){
        matrixFile = "featureMatrixWithoutTexMethods.txt";
        Size size(26, 66);
        resize(data,data, size);
        experimentationPath += "/NotAllTextureMethodsResults/";
    }else{
        experimentationPath += "/AllTextureMethodsResults/";
    }
    
    cout << data.rows << " x " << data.cols << endl;
    
    ifstream file("/Users/kolibrid111/Documents/Universidad/Cuarto/TFG/Matrix/" + matrixFile);
    
    
    // Almacenamos los datos en la matriz. Cada fila corresponde a una muestra
    // de la marposa, y cada columna a una característica.
    while(getline(file, line))
    {
        vector<float>   lineData;
        stringstream  lineStream(line);
        
        float value;
        int j = 0;
        // Read an integer at a time from the line
        while(lineStream >> value)
        {
            data.at<float>(i,j) = value;
            j++;
        }
        cout << j << endl;
        i++;
    }
    
    // Algoritmo de K-means
    kmeans(data, clusterCount, labels,
           TermCriteria( TermCriteria::MAX_ITER, 1000, 0),
           3, KMEANS_RANDOM_CENTERS, centers);
    
    if(HAS_ALL_TEXTURE_METHODS)
        cout << "\nResultados de la clasificación con kmeans para " << clusterCount << " clusters:" << endl;
    else
        cout << "\nResultados de la clasificación con kmeans para " << clusterCount << " clusters sin utilizar todos los métodos de extracción de características de tetxura:" << endl;
    
    cout << "Cluster\t\Imagen" << endl;
    
    // Esta es la dirección donde se almacenaban los resultados del procesamiento de imágenes. Se van a copiar las
    // imágenes de las alas segmentadas y copiar en sus respectivas carpetas del directorio de experimentación.
    string base_url = "/Users/kolibrid111/Documents/Universidad/Cuarto/TFG//Resultados/";
    string folder;
    string url;
    string url_write;
    Mat imagen;
    
    // Almacenamos los resultados de las clasificaciones.
    for (int i = 0; i < 26; i++){
        folder = image[i];
        folder.erase(remove_if(folder.begin(), folder.end(), [](char c) { return !isalpha(c); } ), folder.end());
        folder[0] = toupper(folder[0]);
        
        url = base_url + folder + "/" + image[i] + "/3_5_src_segmented.png";
        
        imagen = imread(url, 1);
        
        url_write = experimentationPath + "cluster" + to_string(labels.at<int>(i, 0) + 1);
        
        url_write += "/" + image[i] + ".png";
        
        imwrite(url_write, imagen);
        
        cout << labels.at<int>(i, 0) << " \t\t" << image[i] << endl;
    }
    
    
}
