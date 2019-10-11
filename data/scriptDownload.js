function main(){
    var rows = document.getElementsByTagName("a");
    var taille = document.getElementsByTagName("a").length;
    var i;
    var link;
    var downloads=new Array();
    for (i=0;i<taille;i++){
    if (rows[i].hasAttribute("href")) 
    {
        link = rows[i].getAttribute("href");
        
        if (link.match(/(bvh)$/)) {
          console.log("href  =>"+link);
            rows[i].setAttribute("dowload", "");
            rows[i].click();
            downloads.push(link);
            
        }
    }
    }

    }
main()
