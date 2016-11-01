
% Autor: Íñigo Alonso Ruiz - 2016


close all; % Se cierran ventanas abiertas en cada ejecución
dinfo = dir('./RectGrabber'); % Path de las imágenes
imagenes=dinfo(:,1);
i=4;
peopleDetector = vision.PeopleDetector;  % De donde se saca la medida (descriptor HOG)

% Matrices del filtro de kalman de nuestro sistema, explicadas en la memoria
Q=[(50/15)^ 2 0 1/15 0; 0 (7/15)^2 0 1/15; 1/15 0 (50)^2 0 ; 0 1/15 0 (7)^2];
A=[1 0 1/15 0; 0 1 0 1/15; 0 0 1 0; 0 0 0 1];
C=[1 0 0 0 ; 0 1 0 0];
R=[8^2 0; 0 8^2];


% Lo primero es encontrar a la persona para poder dar los valores al estado
% inicial
bboxes=[];
personasEnImagen=0;
while personasEnImagen == 0
    imagen=imread(strcat('./RectGrabber/',imagenes(i).name));
    [bboxes, scores] = step(peopleDetector, imagen); % Detect people
    personasEnImagen = size(scores,1);
    % Mientras no se ha encontrado a una personas, siguies pasando im�genes
    if personasEnImagen > 0
        imagen = insertObjectAnnotation(imagen, 'rectangle', bboxes , 'medida');
    end
    i = i+2;
end

%Estado inicial a partir de la imagen y varianza seg�n el detector de HOG
updateEstado = [bboxes(1,1) + bboxes(1,3)/2; bboxes(1,2) + bboxes(1,4)/2;0;0];
updateVarianza =  [8^2 0 1/15 0; 0 8^2 0 1/15; 1/15 0 50^2 0 ; 0 1/15 0 50^2];



%inicializar variables
trazoUpdate=[];
trazoPrediction=[];
trazoMedida=[];
Xvar =0;%en pixeles
Yvar = 0;%en pixeles
framesSinPersonas=0;

% Parar el tracking cuando: hace 25 frames que no hay persona detectacaa o
% la incertidumbre en X � Y es mayor de �250 pixeles
while i < size(imagenes,1) && sqrt(PrediccionVarianza(1,1)) < 250 && sqrt(PrediccionVarianza(2,2)) < 250 &&  framesSinPersonas < 25

    % PREDICCI�N
    PrediccionEstado=A*updateEstado;
    PrediccionVarianza=A*updateVarianza*A'+Q;

    % Variables necesarias para representaci�n visual
    trazoPrediction=[trazoPrediction; PrediccionEstado(1,1)  PrediccionEstado(2,1)];
    VelipsePrediction = [PrediccionVarianza(1,1) PrediccionVarianza(1,2); PrediccionVarianza(2,1) PrediccionVarianza(2,2)];
    nuElipsePrediction  = [PrediccionEstado(1,1) ; PrediccionEstado(2,1)];

    % Leer siguiente imagen
    imagen=imread(strcat('./RectGrabber/',imagenes(i).name));
    [bboxes, scores] = step(peopleDetector, imagen); % Detect people
    personasEnImagen = size(scores,1);


    % MATCHING
    if personasEnImagen > 0
        framesSinPersonas=0;
        % Si hay dato, ver si hago el update
        S = C * PrediccionVarianza * C' + R;
        Xvar = sqrt(S(1,1));%en pixeles
        Yvar = sqrt(S(2,2));%en pixeles

        medida = [bboxes(1,1) + bboxes(1,3)/2; bboxes(1,2) + bboxes(1,4)/2];
        SalidaPredicha = C * PrediccionEstado; % Predicha con modelo
        residuo = medida - SalidaPredicha;

        %Comprobar si tengo que hacer el update (medida v�lida)
        if (abs(residuo(1,1)) < Xvar && abs(residuo(2,1)) < Yvar)
            %No dato espureo
            % UPDATE
            K = PrediccionVarianza * C' * inv(S);
            updateVarianza = (eye(size(K,1)) - K * C) * PrediccionVarianza;
            updateEstado = PrediccionEstado + K * residuo;
        else
            updateVarianza = PrediccionVarianza;
            updateEstado = PrediccionEstado;
        end

        % Variables necesarias para representaci�n visual
        trazoUpdate=[trazoUpdate;updateEstado(1,1) updateEstado(2,1) ];
        trazoMedida=[trazoMedida; bboxes(1,1) + bboxes(1,3)/2 bboxes(1,2) + bboxes(1,4)/2 ];
        imagen = insertObjectAnnotation(imagen, 'rectangle', bboxes , 'medida');
        ultimabox(1,1) = bboxes(1,1) + bboxes(1,3)/20;
        ultimabox(1,2) = bboxes(1,2) + bboxes(1,4)/20;
        ultimabox(1,3) = bboxes(1,3) - bboxes(1,3)/10;
        ultimabox(1,4) = bboxes(1,4) - bboxes(1,4)/10;
        boxUpdate(1,1) = updateEstado(1,1) - ultimabox(1,3)/2 + 10;
        boxUpdate(1,2) = updateEstado(2,1) - ultimabox(1,4)/2 + 10;
        boxUpdate(1,3) = ultimabox(1,3) - 20; boxUpdate(1,4) = ultimabox(1,4) - 20;
        imagen = insertObjectAnnotation(imagen, 'rectangle', boxUpdate , 'update','Color', 'blue');

    else
        updateEstado = PrediccionEstado;
        updateVarianza = PrediccionVarianza;
        % Si no se ha dectectado perosna, aumentar frames sin persona
        framesSinPersonas=framesSinPersonas+1;
        trazoUpdate=[trazoUpdate; PrediccionEstado(1,1)  PrediccionEstado(2,1) ];

    end

    % Variables necesarias para representaci�n visual
    boxPredition(1,1) = PrediccionEstado(1,1) - ultimabox(1,3)/2;
    boxPredition(1,2) = PrediccionEstado(2,1) - ultimabox(1,4)/2;
    boxPredition(1,3) = ultimabox(1,3); boxPredition(1,4) = ultimabox(1,4);
    VelipseUpdate = [updateVarianza(1,1) updateVarianza(1,2); updateVarianza(2,1) updateVarianza(2,2)];
    nuElipseUpdate = [updateEstado(1,1) ; updateEstado(2,1)];
    imagen = insertObjectAnnotation(imagen, 'rectangle', boxPredition , 'prediction', 'Color', 'red');
    figure, imshow(imagen);
    title('Detected people and detection scores ');
    plotUncertainEllip2D(VelipseUpdate, nuElipseUpdate, 'blue');
    plotUncertainEllip2D(VelipsePrediction, nuElipsePrediction, 'red');
    plot(trazoUpdate(:,1),trazoUpdate(:,2), 'blue');
    plot(trazoPrediction(:,1),trazoPrediction(:,2), 'red');
    plot(trazoMedida(:,1),trazoMedida(:,2), 'yellow');

    w = waitforbuttonpress;

    i = i + 2;

end
