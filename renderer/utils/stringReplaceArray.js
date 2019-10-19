function replaceArray(str, find, replace) {
    for (var i = 0; i < find.length; i++) {
        str = str.replace(find[i], replace[i]);
    }
    return str;
};

export {replaceArray};