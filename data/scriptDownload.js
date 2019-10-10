
var rows = document.getElementById("motion_table").getElementsByTagName("tbody")[0].getElementsByTagName("td");
console.log(rows[0]);
var smth;
for (var i in rows){
    try {
    smth=i.getElementsByTagName("td")[0];
    console.log("smth =>"+smth+"   "+i);

    } catch (error) {
        
    }
    
}

function main(){
    var rows = document.getElementById("motion_table").getElementsByTagName("tbody")[0].getElementsByTagName("a");
    var taille = document.getElementById("motion_table").getElementsByTagName("tbody")[0].getElementsByTagName("a").length;
    var i;
    var link;
    var downloads=new Array();
    for (i=0;i<taille;i++){
    if (rows[i].hasAttribute("href")) 
    {
        link = rows[i].getAttribute("href");
        if (link.match(/.*(\.bvh)$/)) {
            downloads.push(link);
            console.log("link => "+link);
        }
    }
    //else {console.log("no");}
    }
    }
main()
