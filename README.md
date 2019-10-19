# Projet_3A_motion_recognition
Apprentissage automatique pour la reconnaissance d’émotion/de style à partir de séquences de mouvement humain        
## data              
#### Les différentes types de data            
<img alt="" src="https://i.imgur.com/QxML83b.gif" /><img alt="" src="https://i.imgur.com/vfge7DS.gif" />

<img alt="" src=https://i.imgur.com/UvBM1gv.gif />           
          
Emotional Body Motion Database : [Téléchargement](https://1drv.ms/u/s!Apv4Ke1FYz8zgQUBfj2P2jsgOC3z?e=klDEYn) ou [site web](http://ebmdb.tuebingen.mpg.de)            
#### Explication des fichiers .bvh (motion capture)       
Vous pouvez utiliser le logiciel [blender](https://www.blender.org) pour visualiser les données .bvh               
La section ROOT spécifie l'emplacement de l'articulation des hanches dans un espace tridimensionnel. Sous la section ROOT de la structure se trouvent des sections JOINT, chacune contenant des informations spécifiant l'emplacement de l'articulation squelettique par rapport à son articulation parent. Les spécifications de localisation relatives pour une articulation parentale et son articulation enfant permettent de déterminer la longueur de l'os osseux entre les deux articulations. Lorsqu'une articulation n'a pas d'articulation enfant, elle est liée à une section End Site.       
<img alt="" src="http://www.cs.cityu.edu.hk/~howard/Teaching/CS4185-5185-2007-SemA/Group12/Terence/BVH%20file%20(part%201).jpg" />          






