## FACE RECOGNITION FOR (RELATIVELY) UNKNOWN POLITICIANS

This is a tiny project of exploration of how machine learning can be applied to tools that expand the accountability of politicians, and the citizens's knowleadge of the people who serves them, and unbestowed to many, they voted for.
The project uses Tensorflow and Convulotional Networks training in order to identify a minister or a secretary of state of the Spanish Government. 

Some context here:
[The most unknown governmet in the history os Spanish Democracy](http://www.elmundo.es/espana/2018/02/18/5a887a2b22601dee4b8b45fe.html)

The InceptionV3 trained model relies on preprocessed images of politicians obtained from Google Images.

The preprocessing uses OpenCV face recognition to crop the faces that will be part of the training dataset. The dataset is compose of at least 40 cropped images of most important members of the Spanish Excutive Branch:

* Presidente del Gobierno: Mariano Rajoy
* Presidencia y para las Administraciones Territoriales: María Soraya Sáenz de Santamaría Antón
* Agricultura y Pesca, Alimentación y Medio Ambiente: Isabel García Tejerina
* Asuntos Exteriores y Cooperació: Alfonso María Dastis Quecedo
* Defensa: María Dolores de Cospedal García
* Educación, Cultura y Deporte
(y Portavoz del Gobierno): Íñigo Méndez de Vigo y Montojo
* Empleo y Seguridad Social	María: Fátima Báñez García
* Energía, Turismo y Agenda Digital: Álvaro Nadal Belda
* Fomento: Íñigo Joaquín de la Serna Hernáiz
* Hacienda y Función Pública: Cristóbal Ricardo Montoro Romero
* Justicia: Rafael Catalá Polo
* Sanidad, Servicios Sociales e Igualdad: Dolors Montserrat y Montserrat
* Interior: Juan Ignacio Zoido Álvarez
...plus every Secretario de Estado (deputy miniter)

Process:
First, I gather images from google images
Second, detect faces and crop them to reduce noise
Third, manually clean the image dataset
Fourth, training
Fifth, testing.


The model is based on this sources:

https://towardsdatascience.com/training-inception-with-tensorflow-on-custom-images-using-cpu-8ecd91595f26
http://answers.opencv.org/question/90010/opencv-python-face-crop-program/
