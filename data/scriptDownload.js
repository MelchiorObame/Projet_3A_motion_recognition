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
        console.log("href  =>"+link);
        if (link.match(/(bvh)$/)) {
            rows[i].setAttribute('dowload', link);
            rows[i].click();
            downloads.push(link);
            
        }
    }
    }

    }
main()
