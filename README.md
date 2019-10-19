# Projet_3A_motion_recognition
Apprentissage automatique pour la reconnaissance d’émotion/de style à partir de séquences de mouvement humain        
## data              
#### Les différentes types de data            
<img alt="sample1" src="https://i.imgur.com/QxML83b.gif" /><img alt="sample2" src="https://i.imgur.com/vfge7DS.gif" />

<img alt="sample3" src="https://i.imgur.com/UvBM1gv.gif" />           
          
Emotional Body Motion Database : [Téléchargement](https://1drv.ms/u/s!Apv4Ke1FYz8zgQUBfj2P2jsgOC3z?e=klDEYn) ou [site web](http://ebmdb.tuebingen.mpg.de)     
Vous pouvez utiliser le logiciel [blender](https://www.blender.org) pour visualiser les données .bvh         
<img alt="blender" src="" />                                        
#### Explication des fichiers .bvh (motion capture)       
             
La section ROOT spécifie l'emplacement de l'articulation des hanches dans un espace tridimensionnel. Sous la section ROOT de la structure se trouvent des sections JOINT, chacune contenant des informations spécifiant l'emplacement de l'articulation squelettique par rapport à son articulation parent. Les spécifications de localisation relatives pour une articulation parentale et son articulation enfant permettent de déterminer la longueur de l'os osseux entre les deux articulations. Lorsqu'une articulation n'a pas d'articulation enfant, elle est liée à une section End Site.       
<img alt="bvh1" src="http://www.cs.cityu.edu.hk/~howard/Teaching/CS4185-5185-2007-SemA/Group12/Terence/BVH%20file%20(part%201).jpg" />     
La section de mouvement commence par le mot clé "MOTION" sur une ligne distincte. Cette ligne est suivie d'une ligne indiquant le nombre d'images. Cette ligne utilise le mot-clé "Frames:" (les deux points font partie du mot-clé) et un nombre indiquant le nombre d'images ou d'échantillons de mouvement contenus dans le fichier. Après la définition des images, la définition "Frame Time:" indique le taux d'échantillonnage des données. Dans l'exemple de fichier BVH, la fréquence d'échantillonnage est 0,04166667, soit 30 images par seconde, soit la fréquence d'échantillonnage habituelle d'un fichier BVH.         
<img alt="bvh2" src="http://www.cs.cityu.edu.hk/~howard/Teaching/CS4185-5185-2007-SemA/Group12/Terence/BVH%20file%20(part%202).jpg" />     







